-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

-- | The two rows whose shape is not a plain call: 'optimizerStep' returns the
-- updated parameters (Haskell values are immutable), 'trainerRun' takes
-- callbacks.
module RLX.Manual
  ( optimizerStep
  , TrainSummary (..)
  , trainerRun
  ) where

import Control.Exception (throwIO)
import Data.Int (Int32)
import Data.List (sortOn)
import Data.Maybe (listToMaybe)

import Foreign.C.Types (CSize (..))
import Foreign.Marshal.Array (withArray)
import Foreign.Marshal.Utils (withMany)
import Foreign.Ptr (Ptr, nullPtr, plusPtr)
import GHC.Clock (getMonotonicTime)

import RLX.Generated (Optimizer, Trainer, trainerGradNorm, trainerSetLr, trainerStep, trainerSteps)
import RLX.Support

foreign import ccall safe "rlx_optimizer_step" c_rlx_optimizer_step
  :: Ptr () -> Ptr () -> Ptr () -> Ptr () -> Ptr () -> Ptr () -> Ptr () -> CSize -> IO Int32

-- | One step: the parameters updated from their gradients.
optimizerStep :: Optimizer -> [(String, [Float])] -> [(String, [Float])] -> IO [(String, [Float])]
optimizerStep opt params grads = do
  let items = sortOn fst params
  gs <- mapM (\(n, _) -> maybe (throwIO (RlxError ErrType ("optimizerStep: no gradient for '" ++ n ++ "'"))) pure (lookup n grads)) items
  withH opt $ \h ->
    withStrs (map fst items) $ \names ->
      withMany withFloats (map snd items) $ \ps ->
        withMany withFloats gs $ \gps ->
          withArray (nullPtr : ps) $ \pp ->
            withArray (nullPtr : gps) $ \gp ->
              withArray (0 : map (fromIntegral . length . snd) items :: [CSize]) $ \pl -> do
                checkRc =<< c_rlx_optimizer_step h names (pp `plusPtr` 8) (gp `plusPtr` 8) (pl `plusPtr` 8) nullPtr nullPtr (fromIntegral (length items))
                mapM (\((n, v), p) -> (,) n <$> floatsAt p (length v)) (zip items ps)

-- | What 'trainerRun' did.
data TrainSummary = TrainSummary
  { summarySteps :: Int
  , summaryStopped :: Bool
  , summaryFirstLoss :: Maybe Float
  , summaryLastLoss :: Maybe Float
  , summaryMeanLoss :: Maybe Float
  , summaryStepsPerSecond :: Double
  , summaryGradNorm :: Maybe Float
  }
  deriving (Show)

-- | Train @steps@ steps, pulling inputs from @batch step@ ('Nothing': data
-- exhausted). @lr step@ schedules the rate; @onStep step loss@ returning
-- 'False' stops early (called every @every@ steps).
trainerRun
  :: Trainer
  -> Int
  -> (Int -> IO (Maybe [(String, [Float])]))
  -> Maybe (Int -> Float)
  -> Maybe (Int -> Float -> IO Bool)
  -> Int
  -> IO TrainSummary
trainerRun t steps batch lr onStep every' = do
  let every = max every' 1
  start <- trainerSteps t
  t0 <- getMonotonicTime
  let go i acc
        | i >= steps = pure (reverse acc, False)
        | otherwise = do
            let s = start + i
            mb <- batch s
            case mb of
              Nothing -> pure (reverse acc, False)
              Just b -> do
                mapM_ (\f -> trainerSetLr t (f s)) lr
                loss <- trainerStep t b
                keep <- case onStep of
                  Just f | (i + 1) `mod` every == 0 -> f s loss
                  _ -> pure True
                if keep then go (i + 1) (loss : acc) else pure (reverse (loss : acc), True)
  (losses, stopped) <- go 0 []
  t1 <- getMonotonicTime
  g <- trainerGradNorm t
  let n = length losses
  pure
    TrainSummary
      { summarySteps = n
      , summaryStopped = stopped
      , summaryFirstLoss = listToMaybe losses
      , summaryLastLoss = if n > 0 then Just (last losses) else Nothing
      , summaryMeanLoss = if n > 0 then Just (sum losses / fromIntegral n) else Nothing
      , summaryStepsPerSecond = fromIntegral n / max (t1 - t0) 1e-9
      , summaryGradNorm = g
      }

