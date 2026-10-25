-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

{-# LANGUAGE OverloadedStrings #-}

-- | Distributed: each rank is a process; start one per rank with the same
-- peers (here three on loopback; across machines, one per host):
--
-- > rlx-example distributed 1 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
-- > rlx-example distributed 2 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
-- > rlx-example distributed 0 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502
--
-- Rank 0 coordinates: collectives, pipelined inference (a stage shipped per
-- worker), data-parallel training. With no arguments it is a mesh of one.
module Distributed (run) where

import Control.Monad (forM_, when)
import Data.Aeson (Value, object, (.=))
import qualified Data.Aeson as A
import qualified Data.Aeson.Key as K
import qualified Data.Aeson.KeyMap as KM
import Data.Binary.Put (putFloatle, runPut)
import qualified Data.ByteString.Lazy as BL
import Data.List (intercalate)
import Data.Scientific (toRealFloat)
import System.Directory (getTemporaryDirectory, removeFile)
import System.FilePath ((</>))
import Text.Printf (printf)

import qualified RLX as R

perRank :: Int
perRank = 16

at :: String -> Value -> Value
at k (A.Object o) = maybe A.Null id (KM.lookup (K.fromString k) o)
at _ _ = A.Null

floats :: Value -> [Float]
floats (A.Array xs) = [toRealFloat n | A.Number n <- foldr (:) [] xs]
floats _ = []

number :: Value -> Double
number (A.Number n) = toRealFloat n
number _ = 0

splitOn :: Char -> String -> [String]
splitOn c s = case break (== c) s of
  (a, []) -> [a]
  (a, _ : rest) -> a : splitOn c rest

run :: [String] -> IO ()
run args = do
  let (rank, world, peers) = case args of
        (r : w : p : _) -> (read r, read w, Just (splitOn ',' p))
        _ -> (0 :: Int, 1 :: Int, Nothing)
  tmp <- getTemporaryDirectory
  let blob name vs = do
        let path = tmp </> ("rlx-dist-hs-" ++ show rank ++ "-" ++ name)
        BL.writeFile path (runPut (mapM_ putFloatle vs))
        pure path
      options = object (("device" .= ("cpu" :: String)) : maybe [] (\p -> ["peers" .= p]) peers)
  g <- R.groupConnectWith R.defGroupConnect {R.groupConnectOptions = Just options} (fromIntegral rank) (fromIntegral world)
  total <- R.groupAllReduceWith R.defGroupAllReduce {R.groupAllReduceOp = "sum"} g [fromIntegral rank + 1]
  gathered <- R.groupAllGather g [fromIntegral rank + 1]
  shared <- R.groupBroadcast g 0 [if rank == 0 then 42 else 0] -- the root's length
  when (rank == 0) $ do
    r <- R.groupRank g
    wsz <- R.groupWorld g
    leader <- R.groupIsLeader g
    printf "mesh: rank %d of %d, leader: %s\n" r wsz (if leader then "true" else "false" :: String)
    printf "all_reduce sum: %s\n" (intercalate "," (map show total))
    printf "all_gather: %s\n" (intercalate "," (map show gathered))
    printf "broadcast: %s\n" (intercalate "," (map show shared))
  if world == 1
    then putStrLn "a mesh of one: pass rank world peers to start more ranks"
    else
      if rank /= 0
        then do
          _ <- R.groupServeWorkerWith R.defGroupServeWorker {R.groupServeWorkerOptions = Just (object ["steps" .= (2 :: Int)])} g
          r <- R.groupServeTrainer g
          R.groupSend g 0 1 (floats (at "w" (at "params" r))) -- rank 0 compares
        else coordinate g world blob

coordinate :: R.Group -> Int -> (String -> [Float] -> IO FilePath) -> IO ()
coordinate g world blob = do
  let stageW = [[fromIntegral (i * 5 `mod` 7) / 7 - 0.3 | i <- [0 .. 15 :: Int]], [fromIntegral (i * 3 `mod` 5) / 5 - 0.2 | i <- [0 .. 15 :: Int]]]
  caps <- R.groupCollectCaps g
  let names = case caps of
        A.Array xs -> [printf "rank %.0f on %s" (number (at "rank" c)) (case at "platform" c of A.String s -> show s; _ -> "?") | c <- foldr (:) [] xs]
        _ -> []
  printf "workers: %s\n" (intercalate ", " (map (filter (/= '"')) names))
  stage <- R.newGraphWith R.defNewGraph {R.newGraphName = "stage"}
  sx <- R.input stage "x" [1, 4]
  sw <- R.param stage "w" [4, 4]
  R.matmul stage sx sw >>= R.relu stage >>= \y -> R.setOutputs stage [y]
  paths <- mapM (\r -> blob ("w" ++ show r ++ ".bin") (stageW !! ((r - 1) `mod` 2))) [1 .. world - 1]
  forM_ (zip [1 .. world - 1] paths) $ \(r, p) ->
    R.groupShipStage g (fromIntegral r) stage (object ["device" .= ("cpu" :: String), "weights" .= object ["w" .= ("file://" ++ p)]])
  forM_ [[1, -1, 0.5, 2], [0.25, 0.5, -0.75, 1]] $ \xv -> do
    R.groupSendActivation g 1 xv
    y <- R.groupRecvActivation g (fromIntegral (world - 1))
    printf "pipeline: %s\n" (unwords [printf "%.3f" v | v <- y] :: String)

  _ <- R.groupCollectCaps g
  lr <- R.newGraphWith R.defNewGraph {R.newGraphName = "lr"}
  bx <- R.input lr "X" [4, 3]
  by <- R.input lr "y" [4, 1]
  wv <- R.param lr "w" [3, 1]
  d <- R.matmul lr bx wv >>= \p -> R.sub lr p by
  R.mul lr d d >>= \sq -> R.mean lr sq [0, 1] >>= \l -> R.setOutputs lr [l]
  bwd <- R.grad lr [wv]
  let wTrue = [1.5, -2, 0.5] :: [Float]
      n = perRank * world
      xAll = [fromIntegral (i * 37 `mod` 23) / 11.5 - 1 | i <- [0 .. n * 3 - 1]] :: [Float]
      yAll = [sum (zipWith (*) (take 3 (drop (i * 3) xAll)) wTrue) | i <- [0 .. n - 1]]
  ux <- blob "X.bin" xAll
  uy <- blob "y.bin" yAll
  uw <- blob "w0.bin" [0, 0, 0]
  let uri p = "file://" ++ p
      spec r =
        object
          [ "params" .= [object ["name" .= ("w" :: String), "uri" .= uri uw]]
          , "data" .=
              [ object ["input" .= ("X" :: String), "uri" .= uri ux, "elem" .= (3 :: Int), "shard_start" .= (r * perRank), "shard_len" .= perRank]
              , object ["input" .= ("y" :: String), "uri" .= uri uy, "elem" .= (1 :: Int), "shard_start" .= (r * perRank), "shard_len" .= perRank]
              ]
          , "batch" .= (4 :: Int)
          , "lr_per_epoch" .= replicate 40 (0.2 :: Double)
          , "device" .= ("cpu" :: String)
          ]
  forM_ [1 .. world - 1] $ \r -> R.groupShipTrain g (fromIntegral r) bwd (spec r)
  mine <- R.groupRunTrain g bwd (spec 0)
  printf "training loss: %.4f -> %.2e\n" (number (at "first_loss" (at "metrics" mine))) (number (at "last_loss" (at "metrics" mine)))
  let w0 = floats (at "w" (at "params" mine))
  printf "rank 0 w: %s (true: 1.5 -2 0.5)\n" (unwords [printf "%.3f" v | v <- w0] :: String)
  forM_ [1 .. world - 1] $ \r -> do
    theirs <- R.groupRecv g (fromIntegral r) 1
    printf "rank %d w agrees: %s\n" r (if and (zipWith (\a b -> abs (a - b) < 1e-5) theirs w0) then "true" else "false" :: String)
  mapM_ removeFile (paths ++ [ux, uy, uw])

