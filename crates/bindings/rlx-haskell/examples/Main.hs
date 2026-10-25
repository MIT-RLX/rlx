-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

-- | The six examples, one per argument:
--
-- > cabal run rlx-example -- inference [device]
-- > cabal run rlx-example -- training [device]
-- > cabal run rlx-example -- loading
-- > cabal run rlx-example -- backends
-- > cabal run rlx-example -- distributed [rank world peers]
-- > cabal run rlx-example -- models
module Main (main) where

import System.Environment (getArgs)
import System.Exit (die)

import qualified Backends
import qualified Distributed
import qualified Inference
import qualified Loading
import qualified Models
import qualified Training

main :: IO ()
main = do
  args <- getArgs
  case args of
    ("inference" : rest) -> Inference.run rest
    ("training" : rest) -> Training.run rest
    ("loading" : _) -> Loading.run
    ("backends" : _) -> Backends.run
    ("distributed" : rest) -> Distributed.run rest
    ("models" : _) -> Models.run
    _ -> die "usage: rlx-example inference|training|loading|backends|distributed|models [args]"
