-- RLX — versatile ML compiler + runtime.
-- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
-- SPDX-License-Identifier: MIT OR Apache-2.0

{-# LANGUAGE RankNTypes #-}

-- | What "RLX.Generated" stands on: errors, the handle class, and moving
-- values to and from C.
module RLX.Support
  ( Node
  , none
  , RlxErrorKind (..)
  , RlxError (..)
  , TypedTensor (..)
  , Handle (..)
  , withH
  , withHMaybe
  , adopt
  , checkRc
  , withStr
  , withStrMaybe
  , withJson
  , withBoxU64
  , withBoxF32
  , withFloats
  , withBytes
  , withU32s
  , withU64s
  , withI64s
  , withStrs
  , withInputs
  , withTypedInputs
  , rows
  , frows
  , allocaOut
  , allocaSlice
  , peekBool
  , peekU32
  , peekU64
  , peekF32
  , peekF32Maybe
  , takeString
  , takeStringMaybe
  , takeJson
  , takeJsonAs
  , takeF32s
  , takeBytes
  , takeU32s
  , takeOutputs
  , takeTypedOutputs
  , takeTensors
  , floatsAt
  ) where

import Control.Exception (Exception, throwIO)
import Control.Monad (forM)
import Data.Aeson (FromJSON, Value, eitherDecodeStrict, encode)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Data.Int (Int32, Int64)
import Data.List (sortOn)
import Data.Word (Word8, Word32, Word64)
import Foreign.C.String (CString, peekCString, withCString)
import Foreign.C.Types (CBool (..), CFloat (..), CSize (..))
import Foreign.ForeignPtr (ForeignPtr, newForeignPtr, withForeignPtr)
import Foreign.Marshal.Alloc (allocaBytes)
import Foreign.Marshal.Array (peekArray, withArray)
import Foreign.Marshal.Utils (fillBytes, toBool, with, withMany)
import Foreign.Ptr (FunPtr, Ptr, castPtr, nullPtr, plusPtr)
import Foreign.Storable (peek, peekByteOff)

-- | A node of a graph: the id every builder returns.
type Node = Word32

-- | "No node", for an optional node argument.
none :: Node
none = 0xFFFFFFFF

-- | What kind of failure an 'RlxError' is (the @RLX_ERR_*@ status).
data RlxErrorKind = ErrType | ErrRange | ErrRuntime | ErrPanic | ErrNotFound | ErrOther
  deriving (Eq, Show)

-- | A failed rlx call.
data RlxError = RlxError {rlxErrorKind :: RlxErrorKind, rlxErrorMessage :: String}
  deriving (Show)

instance Exception RlxError

-- | A tensor of any dtype: its little-endian bytes and dtype name.
data TypedTensor = TypedTensor {typedData :: [Word8], typedDtype :: String}
  deriving (Eq, Show)

-- | A handle type: freed by its C function when collected.
class Handle h where
  handlePtr :: h -> ForeignPtr ()

withH :: Handle h => h -> (Ptr () -> IO r) -> IO r
withH h = withForeignPtr (handlePtr h)

withHMaybe :: Handle h => Maybe h -> (Ptr () -> IO r) -> IO r
withHMaybe Nothing k = k nullPtr
withHMaybe (Just h) k = withH h k

-- | Wrap a handle rlx returned (in the out-parameter @o@).
adopt :: (ForeignPtr () -> h) -> FunPtr (Ptr () -> IO ()) -> Ptr () -> IO h
adopt wrap release o = do
  p <- peek (castPtr o)
  if p == nullPtr
    then throwIO (RlxError ErrRuntime "rlx returned NULL")
    else wrap <$> newForeignPtr release p

foreign import ccall unsafe "rlx_last_error" c_rlx_last_error :: IO CString
foreign import ccall unsafe "rlx_string_free" c_rlx_string_free :: Ptr () -> IO ()
foreign import ccall unsafe "rlx_f32s_release" c_rlx_f32s_release :: Ptr () -> IO ()
foreign import ccall unsafe "rlx_bytes_release" c_rlx_bytes_release :: Ptr () -> IO ()
foreign import ccall unsafe "rlx_u32s_release" c_rlx_u32s_release :: Ptr () -> IO ()
foreign import ccall unsafe "rlx_outputs_len" c_rlx_outputs_len :: Ptr () -> IO CSize
foreign import ccall unsafe "rlx_outputs_get" c_rlx_outputs_get :: Ptr () -> CSize -> Ptr () -> Ptr () -> IO Int32
foreign import ccall unsafe "rlx_outputs_free" c_rlx_outputs_free :: Ptr () -> IO ()
foreign import ccall unsafe "rlx_typed_outputs_len" c_rlx_typed_outputs_len :: Ptr () -> IO CSize
foreign import ccall unsafe "rlx_typed_outputs_get" c_rlx_typed_outputs_get :: Ptr () -> CSize -> Ptr () -> Ptr () -> Ptr () -> IO Int32
foreign import ccall unsafe "rlx_typed_outputs_free" c_rlx_typed_outputs_free :: Ptr () -> IO ()
foreign import ccall unsafe "rlx_tensor_map_len" c_rlx_tensor_map_len :: Ptr () -> IO CSize
foreign import ccall unsafe "rlx_tensor_map_get" c_rlx_tensor_map_get :: Ptr () -> CSize -> Ptr () -> Ptr () -> Ptr () -> IO Int32
foreign import ccall unsafe "rlx_tensor_map_free" c_rlx_tensor_map_free :: Ptr () -> IO ()

-- | Throw the last error when a status is not RLX_OK.
checkRc :: Int32 -> IO ()
checkRc 0 = pure ()
checkRc rc = do
  msg <- c_rlx_last_error >>= peekCString
  throwIO . RlxError kind $ msg
  where
    kind = case rc of
      -1 -> ErrType
      -2 -> ErrRange
      -3 -> ErrRuntime
      -4 -> ErrPanic
      -5 -> ErrNotFound
      _ -> ErrOther

-- ── arguments ──────────────────────────────────────────────────

withStr :: String -> (Ptr () -> IO r) -> IO r
withStr s k = withCString s (k . castPtr)

withStrMaybe :: Maybe String -> (Ptr () -> IO r) -> IO r
withStrMaybe Nothing k = k nullPtr
withStrMaybe (Just s) k = withStr s k

withJson :: Maybe Value -> (Ptr () -> IO r) -> IO r
withJson Nothing k = k nullPtr
withJson (Just v) k = BS.useAsCString (BL.toStrict (encode v)) (k . castPtr)

withBoxU64 :: Maybe Int -> (Ptr () -> IO r) -> IO r
withBoxU64 Nothing k = k nullPtr
withBoxU64 (Just v) k = with (fromIntegral v :: Word64) (k . castPtr)

withBoxF32 :: Maybe Float -> (Ptr () -> IO r) -> IO r
withBoxF32 Nothing k = k nullPtr
withBoxF32 (Just v) k = with (CFloat v) (k . castPtr)

withList :: (a -> b) -> (forall s. [b] -> (Ptr b -> IO s) -> IO s) -> [a] -> (Ptr () -> IO r) -> IO r
withList _ _ [] k = k nullPtr
withList f arr xs k = arr (map f xs) (k . castPtr)

withFloats :: [Float] -> (Ptr () -> IO r) -> IO r
withFloats = withList CFloat withArray

withBytes :: [Word8] -> (Ptr () -> IO r) -> IO r
withBytes = withList id withArray

withU32s :: [Word32] -> (Ptr () -> IO r) -> IO r
withU32s = withList id withArray

withU64s :: [Int] -> (Ptr () -> IO r) -> IO r
withU64s = withList (fromIntegral :: Int -> Word64) withArray

withI64s :: [Int] -> (Ptr () -> IO r) -> IO r
withI64s = withList (fromIntegral :: Int -> Int64) withArray

withStrs :: [String] -> (Ptr () -> IO r) -> IO r
withStrs [] k = k nullPtr
withStrs xs k = withMany withCString xs $ \ps -> withArray ps (k . castPtr)

rows :: [[Int]] -> Int -> Bool -> [Int]
rows v w splat = concat (zipWith row [0 :: Int ..] v)
  where
    row _ r | length r == w = r
    row _ [x] | splat = replicate w x
    row i r = error ("rlx: row " ++ show i ++ " needs " ++ show w ++ " values, got " ++ show (length r))

frows :: [[Float]] -> Int -> [Float]
frows v w = concat (zipWith row [0 :: Int ..] v)
  where
    row i r
      | length r == w = r
      | otherwise = error ("rlx: row " ++ show i ++ " needs " ++ show w ++ " values, got " ++ show (length r))

-- | Named f32 tensors (sorted by name): names, data pointers, lengths, count.
withInputs :: [(String, [Float])] -> ((Ptr (), Ptr (), Ptr (), CSize) -> IO r) -> IO r
withInputs m k =
  let items = sortOn fst m
   in withStrs (map fst items) $ \names ->
        withMany withFloats (map snd items) $ \ptrs ->
          withArray (nullPtr : ptrs) $ \pd ->
            withArray (0 : map (fromIntegral . length . snd) items :: [CSize]) $ \pl ->
              k (names, pd `plusPtr` 8, pl `plusPtr` 8, fromIntegral (length items))

-- | Named typed tensors: names, data pointers, lengths, dtypes, count.
withTypedInputs :: [(String, TypedTensor)] -> ((Ptr (), Ptr (), Ptr (), Ptr (), CSize) -> IO r) -> IO r
withTypedInputs m k =
  let items = sortOn fst m
   in withStrs (map fst items) $ \names ->
        withStrs (map (typedDtype . snd) items) $ \dtypes ->
          withMany withBytes (map (typedData . snd) items) $ \ptrs ->
            withArray (nullPtr : ptrs) $ \pd ->
              withArray (0 : map (fromIntegral . length . typedData . snd) items :: [CSize]) $ \pl ->
                k (names, pd `plusPtr` 8, pl `plusPtr` 8, dtypes, fromIntegral (length items))

-- ── results ────────────────────────────────────────────────────

-- | A zeroed out-parameter big enough for a pointer or any scalar.
allocaOut :: (Ptr () -> IO r) -> IO r
allocaOut k = allocaBytes 8 $ \p -> fillBytes p 0 8 >> k p

-- | A zeroed @{ptr, len}@ out-parameter.
allocaSlice :: (Ptr () -> IO r) -> IO r
allocaSlice k = allocaBytes 16 $ \p -> fillBytes p 0 16 >> k p

peekBool :: Ptr () -> IO Bool
peekBool o = toBool <$> (peek (castPtr o) :: IO CBool)

peekU32 :: Ptr () -> IO Word32
peekU32 o = peek (castPtr o)

peekU64 :: Ptr () -> IO Int
peekU64 o = fromIntegral <$> (peek (castPtr o) :: IO Word64)

peekF32 :: Ptr () -> IO Float
peekF32 o = (\(CFloat f) -> f) <$> peek (castPtr o)

peekF32Maybe :: Ptr () -> Ptr () -> IO (Maybe Float)
peekF32Maybe o k = do
  has <- peekBool k
  if has then Just <$> peekF32 o else pure Nothing

takeStringMaybe :: Ptr () -> IO (Maybe String)
takeStringMaybe o = do
  p <- peek (castPtr o) :: IO (Ptr ())
  if p == nullPtr
    then pure Nothing
    else do
      s <- peekCString (castPtr p)
      c_rlx_string_free p
      pure (Just s)

takeString :: Ptr () -> IO String
takeString o = maybe "" id <$> takeStringMaybe o

takeJsonAs :: FromJSON a => Ptr () -> IO a
takeJsonAs o = do
  p <- peek (castPtr o) :: IO (Ptr ())
  bytes <- BS.packCString (castPtr p)
  c_rlx_string_free p
  either (throwIO . RlxError ErrOther) pure (eitherDecodeStrict bytes)

takeJson :: Ptr () -> IO Value
takeJson = takeJsonAs

floatsAt :: Ptr () -> Int -> IO [Float]
floatsAt _ 0 = pure []
floatsAt p n = map (\(CFloat f) -> f) <$> peekArray n (castPtr p)

slice :: Ptr () -> IO (Ptr (), Int)
slice o = do
  p <- peekByteOff o 0
  n <- peekByteOff o 8 :: IO CSize
  pure (p, fromIntegral n)

takeF32s :: Ptr () -> IO [Float]
takeF32s o = do
  (p, n) <- slice o
  v <- floatsAt p n
  c_rlx_f32s_release o
  pure v

takeBytes :: Ptr () -> IO [Word8]
takeBytes o = do
  (p, n) <- slice o
  v <- if n == 0 then pure [] else peekArray n (castPtr p)
  c_rlx_bytes_release o
  pure v

takeU32s :: Ptr () -> IO [Word32]
takeU32s o = do
  (p, n) <- slice o
  v <- if n == 0 then pure [] else peekArray n (castPtr p)
  c_rlx_u32s_release o
  pure v

takeOutputs :: Ptr () -> IO [[Float]]
takeOutputs out = do
  o <- peek (castPtr out)
  n <- c_rlx_outputs_len o
  vs <- forM (if n == 0 then [] else [0 .. n - 1]) $ \i ->
    allocaOut $ \d -> allocaOut $ \len -> do
      _ <- c_rlx_outputs_get o i d len
      p <- peek (castPtr d)
      k <- peekU64 len
      floatsAt p k
  c_rlx_outputs_free o
  pure vs

takeTypedOutputs :: Ptr () -> IO [TypedTensor]
takeTypedOutputs out = do
  o <- peek (castPtr out)
  n <- c_rlx_typed_outputs_len o
  vs <- forM (if n == 0 then [] else [0 .. n - 1]) $ \i ->
    allocaOut $ \d -> allocaOut $ \len -> allocaOut $ \dt -> do
      _ <- c_rlx_typed_outputs_get o i d len dt
      p <- peek (castPtr d) :: IO (Ptr Word8)
      k <- peekU64 len
      bytes <- if k == 0 then pure [] else peekArray k p
      name <- peek (castPtr dt) >>= peekCString
      pure (TypedTensor bytes name)
  c_rlx_typed_outputs_free o
  pure vs

takeTensors :: Ptr () -> IO [(String, [Float])]
takeTensors out = do
  m <- peek (castPtr out)
  n <- c_rlx_tensor_map_len m
  vs <- forM (if n == 0 then [] else [0 .. n - 1]) $ \i ->
    allocaOut $ \nm -> allocaOut $ \d -> allocaOut $ \len -> do
      _ <- c_rlx_tensor_map_get m i nm d len
      name <- peek (castPtr nm) >>= peekCString
      p <- peek (castPtr d)
      k <- peekU64 len
      v <- floatsAt p k
      pure (name, v)
  c_rlx_tensor_map_free m
  pure vs
