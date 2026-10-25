-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

{-# LANGUAGE OverloadedStrings #-}

-- | The Haskell binding against the real library; exits non-zero on the
-- first failure.
module Main (main) where

import Control.Exception (try)
import Control.Monad (unless)
import Data.Aeson (object, (.=))
import qualified Data.Aeson as A
import qualified Data.Aeson.KeyMap as KM
import Data.IORef (modifyIORef, newIORef, readIORef)
import System.Directory (getTemporaryDirectory, removeFile)
import System.Exit (exitFailure)
import System.FilePath ((</>))
import System.IO (hPutStrLn, stderr)

import qualified RLX as R

lossGraph :: IO (R.Graph, R.Node)
lossGraph = do
  g <- R.newGraphWith R.defNewGraph {R.newGraphName = "mse"}
  x <- R.input g "x" [1, 2]
  y <- R.input g "y" [1, 2]
  w <- R.param g "w" [2, 2]
  d <- R.matmul g x w >>= \p -> R.sub g p y
  R.mul g d d >>= \sq -> R.mean g sq [0, 1] >>= \l -> R.setOutputs g [l]
  pure (g, w)

main :: IO ()
main = do
  checks <- newIORef (0 :: Int)
  let check ok what = do
        modifyIORef checks (+ 1)
        unless ok $ hPutStrLn stderr ("failed: " ++ what) >> exitFailure
      feed = [("x", [1, 2]), ("y", [0, 0])]

  (g, _) <- lossGraph
  m <- R.newSession >>= \s -> R.sessionCompile s g
  R.isConsumed g >>= \c -> check c "compile consumes"
  R.compiledSetParam m "w" [1, 0, 0, 1]
  R.compiledRun m feed >>= \o -> check (head (head o) == 2.5) "loss 2.5"
  R.compiledDevice m >>= \d -> check (d == "cpu") "device"

  r <- try (R.newGraph >>= \g' -> R.relu g' 42)
  check (case r of Left (R.RlxError R.ErrRange _) -> True; _ -> False) "range error"

  (g2, w) <- lossGraph
  bwd <- R.grad g2 [w] >>= \gg -> R.newSession >>= \s -> R.sessionCompile s gg
  R.compiledSetParam bwd "w" [1, 0, 0, 1]
  R.compiledRun bwd (feed ++ [("d_output", [1])]) >>= \o -> check ((o !! 1) !! 3 == 4) "dw"
  t <- R.clone g2 >>= \c -> R.newTrainerWith R.defNewTrainer {R.newTrainerOptions = Just (object ["optimizer" .= object ["kind" .= ("adamw" :: String), "lr" .= (0.1 :: Double)]])} c ["w"] [("w", [0, 0, 0, 0])]
  summary <- R.trainerRun t 300 (\_ -> pure (Just [("x", [1, 1]), ("y", [1, -1])])) Nothing (Just (\_ l -> pure (l > 1e-6))) 1
  check (maybe False (< 1e-4) (R.summaryLastLoss summary)) "trainer converges"
  opt <- R.newOptimizerWith R.defNewOptimizer {R.newOptimizerKind = Just "sgd", R.newOptimizerOptions = Just (object ["lr" .= (0.5 :: Double)])}
  p <- R.optimizerStep opt [("p", [1, -1])] [("p", [2, -2])]
  check (p == [("p", [0, 0])]) "optimizer"

  tmp <- getTemporaryDirectory
  let path = tmp </> "rlx-hs-test.gguf"
  gw <- R.newGgufWriter
  R.ggufWriterAddF32 gw "w" [4] "F32" [0.5, -1, 2, 0.25]
  R.ggufWriterWriteWith R.defGgufWriterWrite {R.ggufWriterWriteArchitecture = Just "demo"} gw path >>= \n -> check (n == 1) "write"
  f <- R.ggufOpen path
  R.ggufArchitecture f >>= \a -> check (a == Just "demo") "architecture"
  R.ggufTensor f "w" >>= \v -> check (v !! 2 == 2) "tensor"
  R.ggufInfo f "w" >>= \i -> check (case i of A.Object o -> KM.lookup "dtype" o == Just (A.String "F32"); _ -> False) "info"
  removeFile path

  R.devices >>= \ds -> check (head ds == "cpu") "cpu"
  (g3, _) <- lossGraph
  runner <- R.newRunnerWith R.defNewRunner {R.newRunnerPolicy = Just (object ["only" .= ["cpu" :: String]])} g3
  R.runnerSetParam runner "w" [1, 0, 0, 1]
  (dev, outs) <- R.runnerRun runner feed
  check (dev == "cpu" && head (head outs) == 2.5) "runner"
  mesh <- R.groupConnect 0 1
  R.groupAllReduceWith R.defGroupAllReduce {R.groupAllReduceOp = "max"} mesh [2, 3] >>= \v -> check (v !! 1 == 3) "all_reduce"
  n <- readIORef checks
  putStrLn ("test_rlx: " ++ show n ++ " checks passed")
