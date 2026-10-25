<?php
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// What the generated src/Generated.php stands on: the library, errors, the
// handle base, and moving values to and from C.

declare(strict_types=1);

namespace Rlx;

use FFI;
use FFI\CData;

/** "No node", for an optional node argument. */
const NONE = 0xFFFFFFFF;

/** A failed rlx call; `kind` is one of type, range, runtime, panic, not_found. */
class RlxException extends \RuntimeException
{
    public function __construct(public readonly string $kind, string $message)
    {
        parent::__construct($message);
    }
}

/** A tensor of any dtype: its little-endian bytes and dtype name. */
final class TypedTensor
{
    public function __construct(public readonly string $data, public readonly string $dtype)
    {
    }
}

/** The base of every handle class: freed in `__destruct`, or now with `close()`. */
abstract class Handle
{
    protected const FREE = '';
    public ?CData $h = null;

    /** @internal Wrap a handle rlx returned. */
    public static function adopt(CData $h): static
    {
        if (FFI::isNull($h)) {
            throw new RlxException('runtime', 'rlx returned NULL');
        }
        $o = (new \ReflectionClass(static::class))->newInstanceWithoutConstructor();
        $o->h = $h;
        return $o;
    }

    /** Free the handle now (it is unusable afterwards). */
    public function close(): void
    {
        if ($this->h !== null && !FFI::isNull($this->h)) {
            Support::ffi()->{static::FREE}($this->h);
        }
        $this->h = null;
    }

    public function __destruct()
    {
        $this->close();
    }
}

/** @internal */
final class Support
{
    private static ?FFI $ffi = null;
    private static array $gated = [];
    private const KINDS = [-1 => 'type', -2 => 'range', -3 => 'runtime', -4 => 'panic', -5 => 'not_found'];

    /**
     * The library: `RLX_CAPI_LIB` when set, else the one bundled with the
     * package (`native/<os>-<arch>/`), else librlx_capi on the search path.
     */
    public static function ffi(): FFI
    {
        return self::$ffi ??= FFI::cdef(file_get_contents(__DIR__ . '/rlx_php.h'), self::libPath());
    }

    private static function libPath(): string
    {
        $env = getenv('RLX_CAPI_LIB');
        if ($env !== false && $env !== '') {
            return $env;
        }
        [$os, $file] = match (PHP_OS_FAMILY) {
            'Darwin' => ['macos', 'librlx_capi.dylib'],
            'Windows' => ['windows', 'rlx_capi.dll'],
            default => ['linux', 'librlx_capi.so'],
        };
        $arch = preg_match('/^(arm64|aarch64)$/i', php_uname('m')) ? 'arm64' : 'x64';
        $bundled = dirname(__DIR__) . "/native/{$os}-{$arch}/{$file}";
        return is_file($bundled) ? $bundled : $file;
    }

    /** A function behind a feature, declared on first use. */
    public static function gated(string $decl): FFI
    {
        return self::$gated[$decl] ??= FFI::cdef($decl, self::libPath());
    }

    public static function check(int $rc): void
    {
        if ($rc !== 0) {
            throw new RlxException(self::KINDS[$rc] ?? 'other', self::ffi()->rlx_last_error());
        }
    }

    // ── arguments ──────────────────────────────────────────────

    public static function json(mixed $v): ?string
    {
        return $v === null ? null : json_encode($v, JSON_THROW_ON_ERROR | JSON_PRESERVE_ZERO_FRACTION | JSON_UNESCAPED_SLASHES);
    }

    /** A C array of `$type` from `$values`, kept alive in `$k`. */
    public static function arr(array &$k, string $type, array $values): ?CData
    {
        $n = count($values);
        if ($n === 0) {
            return null;
        }
        $a = self::ffi()->new("{$type}[$n]");
        $i = 0;
        foreach ($values as $v) {
            $a[$i++] = $v;
        }
        $k[] = $a;
        return $a;
    }

    public static function bytes(array &$k, string $s): ?CData
    {
        $n = strlen($s);
        if ($n === 0) {
            return null;
        }
        $a = self::ffi()->new("uint8_t[$n]");
        FFI::memcpy($a, $s, $n);
        $k[] = $a;
        return $a;
    }

    public static function box(array &$k, string $type, int|float|null $v): ?CData
    {
        if ($v === null) {
            return null;
        }
        $b = self::ffi()->new($type);
        $b->cdata = $v;
        $k[] = $b;
        return FFI::addr($b);
    }

    public static function str(array &$k, string $s): CData
    {
        $n = strlen($s);
        $a = self::ffi()->new('char[' . ($n + 1) . ']');
        FFI::memcpy($a, $s, $n);
        $a[$n] = "\0";
        $k[] = $a;
        return self::ffi()->cast('char*', FFI::addr($a[0]));
    }

    public static function strs(array &$k, ?array $v): ?CData
    {
        if ($v === null || count($v) === 0) {
            return null;
        }
        $n = count($v);
        $a = self::ffi()->new("char*[$n]");
        $i = 0;
        foreach ($v as $s) {
            $a[$i++] = self::str($k, (string) $s);
        }
        $k[] = $a;
        return $a;
    }

    public static function rows(array $rows, int $w, bool $splat): array
    {
        $out = [];
        foreach (array_values($rows) as $i => $r) {
            if (count($r) === $w) {
                array_push($out, ...array_values($r));
            } elseif ($splat && count($r) === 1) {
                array_push($out, ...array_fill(0, $w, array_values($r)[0]));
            } else {
                throw new RlxException('type', sprintf('row %d needs %d values, got %d', $i, $w, count($r)));
            }
        }
        return $out;
    }

    /** [names, data pointers, lengths, n] for named f32 tensors. */
    public static function inputs(array &$k, array $m): array
    {
        ksort($m, SORT_STRING);
        $n = count($m);
        $data = self::ffi()->new('float*[' . max($n, 1) . ']');
        $lens = self::ffi()->new('size_t[' . max($n, 1) . ']');
        $i = 0;
        foreach ($m as $v) {
            $a = self::arr($k, 'float', $v);
            $data[$i] = $a === null ? null : self::ffi()->cast('float*', $a);
            $lens[$i++] = count($v);
        }
        $k[] = $data;
        $k[] = $lens;
        return [self::strs($k, array_map('strval', array_keys($m))), $data, $lens, $n];
    }

    /** [names, data pointers, lengths, dtypes, n] for named TypedTensors. */
    public static function typedInputs(array &$k, array $m): array
    {
        ksort($m, SORT_STRING);
        $n = count($m);
        $data = self::ffi()->new('uint8_t*[' . max($n, 1) . ']');
        $lens = self::ffi()->new('size_t[' . max($n, 1) . ']');
        $i = 0;
        foreach ($m as $t) {
            $a = self::bytes($k, $t->data);
            $data[$i] = $a === null ? null : self::ffi()->cast('uint8_t*', $a);
            $lens[$i++] = strlen($t->data);
        }
        $k[] = $data;
        $k[] = $lens;
        $dtypes = self::strs($k, array_map(fn ($t) => $t->dtype, array_values($m)));
        return [self::strs($k, array_map('strval', array_keys($m))), $data, $lens, $dtypes, $n];
    }

    // ── results ────────────────────────────────────────────────

    public static function takeString(CData $o): string
    {
        return self::takeStringOrNull($o) ?? '';
    }

    public static function takeStringOrNull(CData $o): ?string
    {
        if (FFI::isNull($o)) {
            return null;
        }
        $s = FFI::string($o);
        self::ffi()->rlx_string_free($o);
        return $s;
    }

    public static function decode(string $s): mixed
    {
        return json_decode($s, true, 512, JSON_THROW_ON_ERROR);
    }

    private static function floats(CData $p, int $n): array
    {
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = $p[$i];
        }
        return $out;
    }

    public static function takeF32s(CData $o): array
    {
        $v = $o->len ? self::floats($o->ptr, $o->len) : [];
        self::ffi()->rlx_f32s_release(FFI::addr($o));
        return $v;
    }

    public static function takeBytes(CData $o): string
    {
        $v = $o->len ? FFI::string($o->ptr, $o->len) : '';
        self::ffi()->rlx_bytes_release(FFI::addr($o));
        return $v;
    }

    public static function takeU32s(CData $o): array
    {
        $v = $o->len ? self::floats($o->ptr, $o->len) : [];
        self::ffi()->rlx_u32s_release(FFI::addr($o));
        return $v;
    }

    public static function takeOutputs(CData $o): array
    {
        $f = self::ffi();
        $d = $f->new('float*');
        $len = $f->new('size_t');
        $out = [];
        for ($i = 0, $n = $f->rlx_outputs_len($o); $i < $n; $i++) {
            $f->rlx_outputs_get($o, $i, FFI::addr($d), FFI::addr($len));
            $out[] = $len->cdata ? self::floats($d, $len->cdata) : [];
        }
        $f->rlx_outputs_free($o);
        return $out;
    }

    public static function takeTypedOutputs(CData $o): array
    {
        $f = self::ffi();
        $d = $f->new('uint8_t*');
        $len = $f->new('size_t');
        $dt = $f->new('char*');
        $out = [];
        for ($i = 0, $n = $f->rlx_typed_outputs_len($o); $i < $n; $i++) {
            $f->rlx_typed_outputs_get($o, $i, FFI::addr($d), FFI::addr($len), FFI::addr($dt));
            $out[] = new TypedTensor($len->cdata ? FFI::string($d, $len->cdata) : '', FFI::string($dt));
        }
        $f->rlx_typed_outputs_free($o);
        return $out;
    }

    public static function takeTensors(CData $m): array
    {
        $f = self::ffi();
        $name = $f->new('char*');
        $d = $f->new('float*');
        $len = $f->new('size_t');
        $out = [];
        for ($i = 0, $n = $f->rlx_tensor_map_len($m); $i < $n; $i++) {
            $f->rlx_tensor_map_get($m, $i, FFI::addr($name), FFI::addr($d), FFI::addr($len));
            $out[FFI::string($name)] = $len->cdata ? self::floats($d, $len->cdata) : [];
        }
        $f->rlx_tensor_map_free($m);
        return $out;
    }
}
