-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

{-# LANGUAGE OverloadedStrings #-}

-- | Loading: a GGUF file (written here, f32 and quantized) and a safetensors
-- file into a compiled graph, and quantizing in memory.
module Loading (run) where

import Data.Aeson (object, (.=))
import qualified Data.Aeson as A
import Data.Binary.Put (putFloatle, putWord64le, runPut)
import qualified Data.ByteString.Char8 as BC
import qualified Data.ByteString.Lazy as BL
import Data.Maybe (fromJust)
import System.Directory (getTemporaryDirectory, removeFile)
import System.FilePath ((</>))
import Text.Printf (printf)

import qualified RLX as R

run :: IO ()
run = do
  tmp <- getTemporaryDirectory
  let w = [fromIntegral (i `mod` 9) / 4 - 1 | i <- [0 .. 63 :: Int]] :: [Float]
      gguf = tmp </> "rlx-example-hs.gguf"
  writer <- R.newGgufWriter
  R.ggufWriterAddF32 writer "proj.weight" [8, 8] "F32" w
  R.ggufWriterAddF32 writer "proj.weight_q8" [8, 8] "Q8_0" w
  _ <- R.ggufWriterWriteWith R.defGgufWriterWrite {R.ggufWriterWriteArchitecture = Just "demo", R.ggufWriterWriteMetadata = Just (object ["demo.layers" .= (1 :: Int)])} writer gguf
  f <- R.ggufOpen gguf
  R.ggufArchitecture f >>= printf "gguf architecture: %s\n" . maybe "" id
  R.ggufTensorNames f >>= printf "gguf tensors: %s\n" . unwords
  exact <- R.ggufTensor f "proj.weight"
  q8 <- R.ggufTensor f "proj.weight_q8"
  printf "gguf f32 exact: %s q8 max|Δ|: %.4f\n" (if exact == w then "true" else "false" :: String) (maximum (zipWith (\a b -> abs (a - b)) exact q8))

  g <- R.newGraphWith R.defNewGraph {R.newGraphName = "proj"}
  x <- R.input g "x" [1, 8]
  p <- R.param g "proj.weight" [8, 8]
  R.matmul g x p >>= \y -> R.setOutputs g [y]
  c <- R.newSession >>= \s -> R.sessionCompile s g
  R.compiledSetParam c "proj.weight" exact
  [y] <- R.compiledRun c [("x", replicate 8 1)]
  printf "gguf -> graph: %s\n" (unwords [printf "%.2f" v | v <- take 3 y] :: String)

  let st = tmp </> "rlx-example-hs.safetensors"
      header0 = "{\"proj.weight\":{\"dtype\":\"F32\",\"shape\":[8,8],\"data_offsets\":[0,256]}}"
      header = header0 ++ replicate ((8 - length header0 `mod` 8) `mod` 8) ' '
  BL.writeFile st (runPut (putWord64le (fromIntegral (length header))) <> BL.fromStrict (BC.pack header) <> runPut (mapM_ putFloatle w))
  loaded <- R.loadWeights "safetensors" st
  printf "safetensors: [proj.weight] exact: %s\n" (if fromJust (lookup "proj.weight" loaded) == w then "true" else "false" :: String)

  packed <- R.quantize w "Q8_0"
  back <- R.dequantizeWith R.defDequantize {R.dequantizeCount = Just (length w)} packed "Q8_0"
  printf "quantize q8_0: %d -> %d bytes, round trip %d values\n" (length w * 4) (length packed) (length back)
  removeFile gguf
  removeFile st
  R.argmax [0.1, 2, 0.3] >>= printf "argmax: %d\n"
  where
    _unused = A.Null
