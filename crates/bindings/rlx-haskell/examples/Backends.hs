-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

{-# LANGUAGE OverloadedStrings #-}

-- | Picking a backend: what this build has, what suits a graph, and the ways
-- to choose — by name, by cost model, by policy, by benchmark, per run.
module Backends (run) where

import Control.Monad (forM_)
import Data.Aeson (object, (.=))
import qualified Data.Aeson as A
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Text as T
import Text.Printf (printf)

import qualified RLX as R

model :: IO R.Graph
model = do
  g <- R.newGraphWith R.defNewGraph {R.newGraphName = "mlp"}
  x <- R.input g "x" [4, 16]
  w <- R.param g "w" [16, 16]
  R.matmul g x w >>= R.tanh g >>= \y -> R.setOutputs g [y]
  pure g

run :: IO ()
run = do
  let w = [fromIntegral (i `mod` 5) / 10 - 0.2 | i <- [0 .. 255 :: Int]]
      feed = [("x", [fromIntegral (i `mod` 7) / 7 | i <- [0 .. 63 :: Int]])]
  available <- R.devices
  printf "devices: %s\n" (unwords available)
  R.isAvailable "metal" >>= \m -> printf "metal available: %s\n" (if m then "true" else "false" :: String)
  model >>= R.fastestDeviceFor >>= printf "fastest for graph: %s\n"
  firstOut <- model >>= \g -> R.newSession >>= \s -> R.sessionCompile s g
  R.compiledSetParam firstOut "w" w
  [reference] <- R.compiledRun firstOut feed
  forM_ available $ \device -> do
    c <- model >>= \g -> R.newSessionWith R.defNewSession {R.newSessionDevice = device} >>= \s -> R.sessionCompile s g
    R.compiledSetParam c "w" w
    [y] <- R.compiledRun c feed
    printf "session %s: max|Δ| vs first = %.2e\n" device (maximum (zipWith (\a b -> abs (a - b)) y reference))
  flex <- R.newFlexibleSessionWith R.defNewFlexibleSession {R.newFlexibleSessionPolicy = Just (object ["prefer" .= ["metal", "cpu" :: String]])}
  model >>= R.flexibleSessionCompile flex >>= R.compiledDevice >>= printf "flexible session picked: %s\n"
  runner <- model >>= R.newRunner
  R.runnerSetParam runner "w" w
  R.runnerWarmAll runner >>= printf "runner warmed: %s\n" . unwords
  rows <- R.runnerBenchmark runner feed
  case rows of
    A.Array xs -> forM_ xs $ \r -> case r of
      A.Object o -> printf "  bench %s: exec_ms=%s\n" (str (KM.lookup "device" o)) (num (KM.lookup "exec_ms" o))
      _ -> pure ()
    _ -> pure ()
  R.runnerRun runner feed >>= printf "runner ran on: %s\n" . fst
  R.runnerRunTry runner ["metal", "cpu"] feed >>= printf "run_try chain: %s\n" . fst
  router <- model >>= R.newRouter
  R.routerSetParam router "w" w
  R.routerRunChain router feed >>= printf "router ran on: %s\n" . fst
  where
    str (Just (A.String s)) = T.unpack s
    str _ = "?"
    num (Just (A.Number n)) = show n
    num _ = "?"
