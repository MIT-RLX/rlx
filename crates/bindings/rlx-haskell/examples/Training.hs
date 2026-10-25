-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

{-# LANGUAGE OverloadedStrings #-}

-- | Training: fit y = x·w_true with the Trainer, checkpoint and resume it,
-- then the same fit by hand with grad and an Optimizer.
module Training (run) where

import Data.Aeson (object, (.=))
import qualified Data.Aeson as A
import qualified Data.Aeson.KeyMap as KM
import Data.Maybe (fromJust)
import System.Directory (getTemporaryDirectory, removeFile)
import System.FilePath ((</>))
import Text.Printf (printf)

import qualified RLX as R

wTrue :: [Float]
wTrue = [2, -1, 0.5]

batch :: Int
batch = 8

dataAt :: Int -> [(String, [Float])]
dataAt step =
  let xs = [[fromIntegral ((step * 31 + i * 7 + k * 3) `mod` 17) / 8.5 - 1 | k <- [0 .. 2]] | i <- [0 .. batch - 1]]
   in [("x", concat xs), ("y", [sum (zipWith (*) r wTrue) | r <- xs])]

fmt :: [Float] -> String
fmt v = unwords [printf "%.3f" e | e <- v]

run :: [String] -> IO ()
run args = do
  let device = case args of (d : _) -> d; _ -> "cpu"
  g <- R.newGraphWith R.defNewGraph {R.newGraphName = "regression"}
  x <- R.input g "x" [batch, 3]
  y <- R.input g "y" [batch, 1]
  w <- R.param g "w" [3, 1]
  d <- R.matmul g x w >>= \p -> R.sub g p y
  loss <- R.mul g d d >>= \sq -> R.mean g sq [0, 1]
  R.setOutputs g [loss]

  let opts = object ["device" .= device, "optimizer" .= object ["kind" .= ("adamw" :: String), "lr" .= (0.1 :: Double)]]
  trainer <- R.clone g >>= \g' -> R.newTrainerWith R.defNewTrainer {R.newTrainerOptions = Just opts} g' ["w"] [("w", [0, 0, 0])]
  first <- R.trainerEvaluate trainer (dataAt 0)
  _ <- R.trainerRun trainer 400 (pure . Just . dataAt)
    (Just (\s -> 0.1 * 0.5 * (1 + cos (pi * fromIntegral s / 400))))
    (Just (\_ l -> pure (l > 1e-6))) 1
  final <- R.trainerEvaluate trainer (dataAt 0)
  steps <- R.trainerSteps trainer
  printf "trainer loss: %.4f -> %.6f in %d steps\n" first final steps
  params <- R.trainerParams trainer
  printf "trainer w: %s\n" (fmt (fromJust (lookup "w" params)))

  tmp <- getTemporaryDirectory
  let path = tmp </> "rlx-example-train-hs.gguf"
  written <- R.trainerSave trainer path
  restored <- R.clone g >>= \g' -> R.newTrainerWith R.defNewTrainer {R.newTrainerOptions = Just (object ["device" .= device])} g' ["w"] [("w", [0, 0, 0])]
  info <- R.trainerLoad restored path
  let step = case info of A.Object o -> KM.lookup "step" o; _ -> Nothing
  printf "checkpoint: %d tensors, restored step %s\n" written (maybe "?" show' step)
  removeFile path
  _ <- R.trainerFreezeWith R.defTrainerFreeze {R.trainerFreezeNames = Just ["w"]} restored
  before <- fmt . fromJust . lookup "w" <$> R.trainerParams restored
  _ <- R.trainerStep restored (dataAt 1)
  after <- fmt . fromJust . lookup "w" <$> R.trainerParams restored
  frozen <- R.trainerFrozen restored
  printf "frozen: %s unchanged: %s\n" (show frozen) (if after == before then "true" else "false" :: String)
  _ <- R.trainerUnfreezeWith R.defTrainerUnfreeze restored

  session <- R.newSessionWith R.defNewSession {R.newSessionDevice = device}
  backward <- R.grad g [w] >>= R.sessionCompile session -- outputs [loss, dw]
  opt <- R.newOptimizerWith R.defNewOptimizer {R.newOptimizerKind = Just "sgd", R.newOptimizerOptions = Just (object ["lr" .= (0.2 :: Double)])}
  let loop :: Int -> [(String, [Float])] -> Float -> IO ([(String, [Float])], Float)
      loop 300 p l = pure (p, l)
      loop s p _ = do
        R.compiledSetParam backward "w" (fromJust (lookup "w" p))
        [[l], dw] <- R.compiledRun backward (dataAt s ++ [("d_output", [1])])
        p' <- R.optimizerStep opt p [("w", dw)]
        loop (s + 1) p' l
  (final', l') <- loop 0 [("w", [0, 0, 0])] 0
  printf "manual loss: %.6f w: %s\n" l' (fmt (fromJust (lookup "w" final')))
  where
    show' v = case v of A.Number n -> show (round n :: Int); _ -> "?"
