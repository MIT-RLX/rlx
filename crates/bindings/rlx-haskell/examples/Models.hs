{-# LANGUAGE OverloadedStrings #-}

-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

-- | Models: any provider's model, opened by path or id and run by name. A
-- graph saved as an .rlxp package runs anywhere; a plugin's models (a
-- downstream collection's LLMs, or the reference plugin's demo:count) load
-- through RLX_MODEL_PLUGINS.
module Models (run) where

import Data.Aeson (Value, object, (.=))
import qualified Data.Aeson as A
import qualified Data.Aeson.Key as K
import qualified Data.Aeson.KeyMap as KM
import Data.Scientific (toRealFloat)
import qualified Data.Text as T
import System.Directory (getTemporaryDirectory, removeFile)
import System.FilePath ((</>))
import Text.Printf (printf)

import qualified RLX as R

-- | The items of a JSON array.
elems :: Value -> [Value]
elems (A.Array xs) = foldr (:) [] xs
elems _ = []

str :: String -> Value -> String
str k (A.Object o) | Just (A.String s) <- KM.lookup (K.fromString k) o = T.unpack s
str _ _ = "?"

run :: IO ()
run = do
  g <- R.newGraphWith R.defNewGraph {R.newGraphName = "affine"}
  x <- R.input g "x" [1, 2]
  w <- R.param g "w" [2, 2]
  y <- R.matmul g x w
  R.setOutputs g [y]
  tmp <- getTemporaryDirectory
  let path = tmp </> "rlx-example-haskell.rlxp"
      meta = object ["task" .= ("graph" :: String), "family" .= ("affine" :: String), "outputs" .= ["y" :: String]]
  R.saveModelWith R.defSaveModel {R.saveModelModel = Just meta} g path [("w", [2, 0, 0, 3])]

  m <- R.modelOpenWith R.defModelOpen {R.modelOpenOptions = Just (object ["device" .= ("cpu" :: String)])} path
  info <- R.modelInfo m
  printf "model: %s (%s) on %s\n" (str "family" info) (str "task" info) (str "device" info)
  out <- R.modelRun m [("x", [1, 1])]
  case out of
    (v : _) -> printf "model run: %.3f %.3f\n" (v !! 0) (v !! 1)
    _ -> pure ()
  removeFile path

  lm <- R.modelOpen "demo:count"
  lmInfo <- R.modelInfo lm
  printf "plugin model: %s (%s)\n" (str "family" lmInfo) (str "task" lmInfo)
  text <- R.modelGenerateTextWith R.defModelGenerateText {R.modelGenerateTextOptions = Just (object ["max_tokens" .= (4 :: Int)])} lm "c"
  printf "model text: %s\n" text
  ids <- R.modelGenerateWith R.defModelGenerate {R.modelGenerateMaxTokens = 3} lm [2]
  printf "model ids: %s\n" (unwords (map show ids))

  -- By task: a chat, speech out and back in, embeddings.
  let msgs = A.toJSON [object ["role" .= ("user" :: String), "content" .= ("c" :: String)]]
  reply <- R.modelChatWith R.defModelChat {R.modelChatOptions = Just (object ["max_tokens" .= (4 :: Int)])} lm msgs
  printf "chat: %s\n" reply
  (audio, rate) <- R.modelOpen "demo:voice" >>= \v -> R.modelSynthesize v "hello"
  printf "voice: %d samples at %d Hz\n" (length audio) (fromIntegral rate :: Int)
  ear <- R.modelOpen "demo:speech"
  heard <- R.modelTranscribeWith R.defModelTranscribe {R.modelTranscribeSampleRate = rate} ear audio
  printf "heard: %s\n" heard
  rows <- R.modelOpen "demo:embed" >>= \e -> R.modelEmbed e ["hello", "sky"]
  printf "embed: %s\n" (unwords [show (round (toRealFloat n :: Double) :: Int) | A.Array r <- elems rows, A.Number n <- foldr (:) [] r])
