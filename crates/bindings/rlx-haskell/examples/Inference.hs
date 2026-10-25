-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

-- | Inference: build a graph, compile it for a device, set its weights, run it.
-- A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].
module Inference (run) where

import Text.Printf (printf)

import qualified RLX as R

weights :: Int -> Int -> [Float]
weights n seed = [fromIntegral ((i * 7919 + seed * 104729) `mod` 1000) / 1000 - 0.5 | i <- [0 .. n - 1]]

run :: [String] -> IO ()
run args = do
  let device = case args of (d : _) -> d; _ -> "cpu"
  g <- R.newGraphWith R.defNewGraph {R.newGraphName = "classifier"}
  x <- R.input g "x" [1, 4]
  w1 <- R.param g "w1" [4, 8]
  hidden <- R.matmul g x w1 >>= R.relu g
  w2 <- R.param g "w2" [8, 3]
  out <- R.matmul g hidden w2 >>= R.softmax g
  R.setOutputs g [out]

  session <- R.newSessionWith R.defNewSession {R.newSessionDevice = device}
  model <- R.sessionCompile session g -- consumes g
  R.compiledDevice model >>= printf "compiled on: %s\n"
  _ <- R.compiledSetParams model [("w1", weights 32 1), ("w2", weights 24 2)]
  [probs] <- R.compiledRun model [("x", [0.5, -1, 2, 0.25])]
  printf "probs: %s\n" (unwords [printf "%.4f" p | p <- probs] :: String)
  printf "probs sum: %.4f\n" (sum probs)
  shapes <- R.compiledOutputShapes model
  dtypes <- R.compiledOutputDtypes model
  printf "output shapes: %s dtypes: %s\n" (show shapes) (show dtypes)
