// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CORE-MATH `asinh` / `acosh`'s shared tables (glibc
//! `sysdeps/ieee754/dbl-64/s_asincosh_data.c`, MIT), the bits as compiled.
//! Do not edit.

pub(crate) const R1: [f64; 33] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fef507600000000),
    f64::from_bits(0x3feea4b000000000),
    f64::from_bits(0x3fedfc9800000000),
    f64::from_bits(0x3fed581800000000),
    f64::from_bits(0x3fecb72000000000),
    f64::from_bits(0x3fec199c00000000),
    f64::from_bits(0x3feb7f7600000000),
    f64::from_bits(0x3feae8a000000000),
    f64::from_bits(0x3fea550400000000),
    f64::from_bits(0x3fe9c49200000000),
    f64::from_bits(0x3fe9373800000000),
    f64::from_bits(0x3fe8ace600000000),
    f64::from_bits(0x3fe8258a00000000),
    f64::from_bits(0x3fe7a11400000000),
    f64::from_bits(0x3fe71f7600000000),
    f64::from_bits(0x3fe6a09e00000000),
    f64::from_bits(0x3fe6247e00000000),
    f64::from_bits(0x3fe5ab0800000000),
    f64::from_bits(0x3fe5342c00000000),
    f64::from_bits(0x3fe4bfda00000000),
    f64::from_bits(0x3fe44e0800000000),
    f64::from_bits(0x3fe3dea600000000),
    f64::from_bits(0x3fe371a800000000),
    f64::from_bits(0x3fe306fe00000000),
    f64::from_bits(0x3fe29e9e00000000),
    f64::from_bits(0x3fe2387a00000000),
    f64::from_bits(0x3fe1d48800000000),
    f64::from_bits(0x3fe172b800000000),
    f64::from_bits(0x3fe1130200000000),
    f64::from_bits(0x3fe0b55800000000),
    f64::from_bits(0x3fe059b000000000),
    f64::from_bits(0x3fe0000000000000),
];

pub(crate) const R2: [f64; 33] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3feffa7400000000),
    f64::from_bits(0x3feff4ea00000000),
    f64::from_bits(0x3fefef6200000000),
    f64::from_bits(0x3fefe9da00000000),
    f64::from_bits(0x3fefe45200000000),
    f64::from_bits(0x3fefdecc00000000),
    f64::from_bits(0x3fefd94600000000),
    f64::from_bits(0x3fefd3c200000000),
    f64::from_bits(0x3fefce3e00000000),
    f64::from_bits(0x3fefc8bc00000000),
    f64::from_bits(0x3fefc33a00000000),
    f64::from_bits(0x3fefbdba00000000),
    f64::from_bits(0x3fefb83a00000000),
    f64::from_bits(0x3fefb2bc00000000),
    f64::from_bits(0x3fefad3e00000000),
    f64::from_bits(0x3fefa7c200000000),
    f64::from_bits(0x3fefa24600000000),
    f64::from_bits(0x3fef9cca00000000),
    f64::from_bits(0x3fef975000000000),
    f64::from_bits(0x3fef91d800000000),
    f64::from_bits(0x3fef8c6000000000),
    f64::from_bits(0x3fef86e800000000),
    f64::from_bits(0x3fef817200000000),
    f64::from_bits(0x3fef7bfe00000000),
    f64::from_bits(0x3fef768a00000000),
    f64::from_bits(0x3fef711600000000),
    f64::from_bits(0x3fef6ba400000000),
    f64::from_bits(0x3fef663200000000),
    f64::from_bits(0x3fef60c200000000),
    f64::from_bits(0x3fef5b5200000000),
    f64::from_bits(0x3fef55e400000000),
    f64::from_bits(0x3fef507600000000),
];

pub(crate) const L1: [[f64; 2]; 33] = [
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x0000000000000000),
    ],
    [
        f64::from_bits(0xbd1269e2038315b3),
        f64::from_bits(0x3f962e4eacd40000),
    ],
    [
        f64::from_bits(0xbd23f2558bddfc47),
        f64::from_bits(0x3fa62e3ce7218000),
    ],
    [
        f64::from_bits(0x3d207ea13c34efb5),
        f64::from_bits(0x3fb0a2ab6d3ec000),
    ],
    [
        f64::from_bits(0x3d38f3e77084d3ba),
        f64::from_bits(0x3fb62e4a86d8c000),
    ],
    [
        f64::from_bits(0xbd18d92a005f1a7e),
        f64::from_bits(0x3fbbb9db7062c000),
    ],
    [
        f64::from_bits(0x3d358239e799bfe5),
        f64::from_bits(0x3fc0a2b1a22cc000),
    ],
    [
        f64::from_bits(0xbd3a93fcf5f593b7),
        f64::from_bits(0x3fc3687f0a298000),
    ],
    [
        f64::from_bits(0xbd1db4cac32fd2b5),
        f64::from_bits(0x3fc62e4116b64000),
    ],
    [
        f64::from_bits(0xbd10e65a92ee0f3b),
        f64::from_bits(0x3fc8f409e4df6000),
    ],
    [
        f64::from_bits(0xbd38261383d475f1),
        f64::from_bits(0x3fcbb9d15001c000),
    ],
    [
        f64::from_bits(0xbd3359886207513b),
        f64::from_bits(0x3fce7f9a8c940000),
    ],
    [
        f64::from_bits(0x3d3811f87496ceb7),
        f64::from_bits(0x3fd0a2b052ddb000),
    ],
    [
        f64::from_bits(0x3d34991ec6cb435c),
        f64::from_bits(0x3fd205955ef73000),
    ],
    [
        f64::from_bits(0xbd34581abfeb8927),
        f64::from_bits(0x3fd3687bd9121000),
    ],
    [
        f64::from_bits(0x3d3cab48f6942703),
        f64::from_bits(0x3fd4cb5e8f2b5000),
    ],
    [
        f64::from_bits(0xbd0df2c452fde132),
        f64::from_bits(0x3fd62e4420e20000),
    ],
    [
        f64::from_bits(0x3d26109f4fdb74bd),
        f64::from_bits(0x3fd791292c46a000),
    ],
    [
        f64::from_bits(0xbd36b95fbdac7696),
        f64::from_bits(0x3fd8f40af84e7000),
    ],
    [
        f64::from_bits(0x3d17394fa880cbda),
        f64::from_bits(0x3fda56ed8f865000),
    ],
    [
        f64::from_bits(0xbd150b06a94eccab),
        f64::from_bits(0x3fdbb9d6505b4000),
    ],
    [
        f64::from_bits(0xbd3be2abf0b38989),
        f64::from_bits(0x3fdd1cb91e728000),
    ],
    [
        f64::from_bits(0xbd37d6bf1e34da04),
        f64::from_bits(0x3fde7f9d139e2000),
    ],
    [
        f64::from_bits(0xbd3423c1e14de6ed),
        f64::from_bits(0x3fdfe27db9b0e000),
    ],
    [
        f64::from_bits(0x3d3c46f1a0efbbc2),
        f64::from_bits(0x3fe0a2b25060a800),
    ],
    [
        f64::from_bits(0x3d2834fe4e3e6018),
        f64::from_bits(0x3fe154244482a000),
    ],
    [
        f64::from_bits(0x3d16a03d0f02b650),
        f64::from_bits(0x3fe2059731298800),
    ],
    [
        f64::from_bits(0x3d3d437056526f30),
        f64::from_bits(0x3fe2b707145de000),
    ],
    [
        f64::from_bits(0xbd2a0233728405c5),
        f64::from_bits(0x3fe3687b0e0b2800),
    ],
    [
        f64::from_bits(0xbd24dbdda10d2bf1),
        f64::from_bits(0x3fe419ec5d3f6800),
    ],
    [
        f64::from_bits(0x3d3f7d0a25d154f2),
        f64::from_bits(0x3fe4cb5f9fc02000),
    ],
    [
        f64::from_bits(0x3d315ede4d803b18),
        f64::from_bits(0x3fe57cd28421a800),
    ],
    [
        f64::from_bits(0x3d2ef35793c76730),
        f64::from_bits(0x3fe62e42fefa3800),
    ],
];

pub(crate) const L2: [[f64; 2]; 33] = [
    [
        f64::from_bits(0x0000000000000000),
        f64::from_bits(0x0000000000000000),
    ],
    [
        f64::from_bits(0x3d35abdac3638e99),
        f64::from_bits(0x3f4631ec81e00000),
    ],
    [
        f64::from_bits(0xbd216b8be9bbe239),
        f64::from_bits(0x3f562fd812700000),
    ],
    [
        f64::from_bits(0xbd3364c6315542eb),
        f64::from_bits(0x3f60a25205080000),
    ],
    [
        f64::from_bits(0x3d2734abe459c900),
        f64::from_bits(0x3f662dadc1d00000),
    ],
    [
        f64::from_bits(0x3d30cf8a761431bf),
        f64::from_bits(0x3f6bb9ff94d00000),
    ],
    [
        f64::from_bits(0x3d2da2718eb78708),
        f64::from_bits(0x3f70a2a2def80000),
    ],
    [
        f64::from_bits(0x3d334ada62c59b93),
        f64::from_bits(0x3f7368c0fae40000),
    ],
    [
        f64::from_bits(0x3d3d09ab376682d4),
        f64::from_bits(0x3f762e58e4f80000),
    ],
    [
        f64::from_bits(0xbd23cb7b94329211),
        f64::from_bits(0x3f78f46bd28c0000),
    ],
    [
        f64::from_bits(0xbd2eec5c297c41d0),
        f64::from_bits(0x3f7bb9f831200000),
    ],
    [
        f64::from_bits(0xbd36411b9395d150),
        f64::from_bits(0x3f7e7fff8f300000),
    ],
    [
        f64::from_bits(0xbd31c0e59a43053c),
        f64::from_bits(0x3f80a2c0006e0000),
    ],
    [
        f64::from_bits(0x3d16506596e077b6),
        f64::from_bits(0x3f8205bdb6f00000),
    ],
    [
        f64::from_bits(0x3d3e256bce6faa27),
        f64::from_bits(0x3f836877c86e0000),
    ],
    [
        f64::from_bits(0x3ccbd42467b0c8d1),
        f64::from_bits(0x3f84cb6f55780000),
    ],
    [
        f64::from_bits(0xbd3c4f92132ff0f0),
        f64::from_bits(0x3f862e230e8c0000),
    ],
    [
        f64::from_bits(0xbd380be08bfab390),
        f64::from_bits(0x3f87911440f60000),
    ],
    [
        f64::from_bits(0xbd3f0b1319ceb1f7),
        f64::from_bits(0x3f88f443020a0000),
    ],
    [
        f64::from_bits(0x3d2a65fcfb8de99b),
        f64::from_bits(0x3f8a572dbef40000),
    ],
    [
        f64::from_bits(0x3d14233885d3779c),
        f64::from_bits(0x3f8bb9d449a60000),
    ],
    [
        f64::from_bits(0x3d3f46a59e646edb),
        f64::from_bits(0x3f8d1cb8491c0000),
    ],
    [
        f64::from_bits(0xbd3c3d2f11c11446),
        f64::from_bits(0x3f8e7fd9d2aa0000),
    ],
    [
        f64::from_bits(0x3d27763f78a1e0cc),
        f64::from_bits(0x3f8fe2b6f9780000),
    ],
    [
        f64::from_bits(0x3d3b4c37fc60c043),
        f64::from_bits(0x3f90a2a7c7a50000),
    ],
    [
        f64::from_bits(0xbd15b8a822859be3),
        f64::from_bits(0x3f915412ca860000),
    ],
    [
        f64::from_bits(0xbd3f2d8c9fc06400),
        f64::from_bits(0x3f92059c90050000),
    ],
    [
        f64::from_bits(0xbd3e80e79c20378d),
        f64::from_bits(0x3f92b703f49b0000),
    ],
    [
        f64::from_bits(0x3d368256e4329bdb),
        f64::from_bits(0x3f93688a1a8d0000),
    ],
    [
        f64::from_bits(0x3d37e9741da248c3),
        f64::from_bits(0x3f9419edc7ba0000),
    ],
    [
        f64::from_bits(0x3d2e330dccce602b),
        f64::from_bits(0x3f94cb7034fa0000),
    ],
    [
        f64::from_bits(0x3ce2f32b5d18eefb),
        f64::from_bits(0x3f957cd011870000),
    ],
    [
        f64::from_bits(0xbd1269e2038315b3),
        f64::from_bits(0x3f962e4eacd40000),
    ],
];

pub(crate) const C: [f64; 5] = [
    f64::from_bits(0xbfe0000000000000),
    f64::from_bits(0x3fd5555555555530),
    f64::from_bits(0xbfcfffffffffffa0),
    f64::from_bits(0x3fc99999e33a6366),
    f64::from_bits(0xbfc555559ef9525f),
];

/// `(c0: u16, c1: i16)`.
pub(crate) const B: [(u16, i16); 32] = [
    (301, 27565),
    (7189, 24786),
    (13383, 22167),
    (18923, 19696),
    (23845, 17361),
    (28184, 15150),
    (31969, 13054),
    (35231, 11064),
    (37996, 9173),
    (40288, 7372),
    (42129, 5657),
    (43542, 4020),
    (44546, 2457),
    (45160, 962),
    (45399, -468),
    (45281, -1838),
    (44821, -3151),
    (44032, -4412),
    (42929, -5622),
    (41522, -6786),
    (39825, -7905),
    (37848, -8982),
    (35602, -10020),
    (33097, -11020),
    (30341, -11985),
    (27345, -12916),
    (24115, -13816),
    (20661, -14685),
    (16989, -15526),
    (13107, -16339),
    (9022, -17126),
    (4740, -17889),
];
