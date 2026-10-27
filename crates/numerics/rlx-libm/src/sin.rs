// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! glibc's double `sin` / `cos` (IBM Accurate Mathematical Library,
//! `sysdeps/ieee754/dbl-64/s_sin.c`) as x86-64 runs it on an FMA CPU —
//! `__sin_fma` / `__cos_fma`, gcc contracting every product whose uses are all
//! adds/subtracts (a product with several such uses is fused into each) — with the
//! huge-argument reduction `__branred` (`branred.c`, built without FMA).

use crate::fma;

const S1: f64 = f64::from_bits(0xbfc5555555555555);
const S2: f64 = f64::from_bits(0x3f81111111110ece);
const S3: f64 = f64::from_bits(0xbf2a01a019db08b8);
const S4: f64 = f64::from_bits(0x3ec71de27b9a7ed9);
const S5: f64 = f64::from_bits(0xbe5addffc2fcdf59);
const BIG: f64 = f64::from_bits(0x42c8000000000000);
const HP0: f64 = f64::from_bits(0x3ff921fb54442d18);
const HP1: f64 = f64::from_bits(0x3c91a62633145c07);
const MP1: f64 = f64::from_bits(0x3ff921fb58000000);
const MP2: f64 = f64::from_bits(0xbe4dde973c000000);
const PP3: f64 = f64::from_bits(0xbc8cb3b398000000);
const PP4: f64 = f64::from_bits(0xbacd747f23e32ed7);
const HPINV: f64 = f64::from_bits(0x3fe45f306dc9c883);
const TOINT: f64 = f64::from_bits(0x4338000000000000);
const SN3: f64 = f64::from_bits(0xbfc5555555555515); // -1.66666666666664880952546298448555E-01
const SN5: f64 = f64::from_bits(0x3f811110e829872f); // 8.33333214285722277379541354343671E-03
const CS2: f64 = f64::from_bits(0x3fe0000000000000); // 4.99999999999999999999950396842453E-01
const CS4: f64 = f64::from_bits(0xbfa5555555555535); // -4.16666666666664434524222570944589E-02
const CS6: f64 = f64::from_bits(0x3f56c16bedd9e239); // 1.38888874007937613028114285595617E-03

const TOVERP: [f64; 75] = [
    10680707.0, 7228996.0, 1387004.0, 2578385.0, 16069853.0, 12639074.0, 9804092.0, 4427841.0,
    16666979.0, 11263675.0, 12935607.0, 2387514.0, 4345298.0, 14681673.0, 3074569.0, 13734428.0,
    16653803.0, 1880361.0, 10960616.0, 8533493.0, 3062596.0, 8710556.0, 7349940.0, 6258241.0,
    3772886.0, 3769171.0, 3798172.0, 8675211.0, 12450088.0, 3874808.0, 9961438.0, 366607.0,
    15675153.0, 9132554.0, 7151469.0, 3571407.0, 2607881.0, 12013382.0, 4155038.0, 6285869.0,
    7677882.0, 13102053.0, 15825725.0, 473591.0, 9065106.0, 15363067.0, 6271263.0, 9264392.0,
    5636912.0, 4652155.0, 7056368.0, 13614112.0, 10155062.0, 1944035.0, 9527646.0, 15080200.0,
    6658437.0, 6231200.0, 6832269.0, 16767104.0, 5075751.0, 3212806.0, 1398474.0, 7579849.0,
    6349435.0, 12618859.0, 4703257.0, 12806093.0, 14477321.0, 2786137.0, 12875403.0, 9837734.0,
    14528324.0, 13719321.0, 343717.0,
];

/// glibc `__sincostab` (`sysdeps/ieee754/dbl-64/sincostab.c`): per table point
/// `sn, ssn, cs, ccs` as IEEE bits.
const SINCOSTAB: [u64; 440] = [
    0x0000000000000000,
    0x0000000000000000,
    0x3ff0000000000000,
    0x0000000000000000,
    0x3f7fffeaaaaeeeef,
    0xbc1e45e2ec67b77c,
    0x3fefffc000155552,
    0x3c8f4a01a0196dae,
    0x3f8fffaaaaeeeed5,
    0xbc02ab639a9f0777,
    0x3fefff000155549f,
    0x3c828a28a03a5ef3,
    0x3f97ff7001033255,
    0x3bfefe2b51527336,
    0x3feffdc006bff7e6,
    0x3c8ae6dae86977bd,
    0x3f9ffeaaaeeee86f,
    0xbc3cd406fb224ae2,
    0x3feffc00155527d3,
    0xbc83b54492d89b5b,
    0x3fa3feb2b12d45d5,
    0x3c34ec54203d1c11,
    0x3feff9c03414a7ba,
    0x3c6991f4be6c59bf,
    0x3fa7fdc01032fba9,
    0xbc4599bdf46e997a,
    0x3feff7006bfdf99f,
    0xbc78b3b560648d5f,
    0x3fabfc6d78586dac,
    0x3c18e4fd03dbf236,
    0x3feff3c0c8103a31,
    0x3c74856dbddc0e66,
    0x3faffaaaeeed4edb,
    0xbc42d16d32684b69,
    0x3feff0015549f4d3,
    0x3c8328387b99426f,
    0x3fb1fc343d808bef,
    0xbc5f3d32e6f3be4f,
    0x3fefebc222a8ef9f,
    0x3c57934934f54c77,
    0x3fb3facb12d1755b,
    0xbc5921915299468c,
    0x3fefe7034129ef6f,
    0xbc6cbf4337c96f97,
    0x3fb5f911fd10b737,
    0xbc50184f02be9102,
    0x3fefe1c4c3c873eb,
    0xbc35a9c9057c4a02,
    0x3fb7f701032550e4,
    0x3c3afc2d1800501a,
    0x3fefdc06bf7e6b9b,
    0x3c831902b535f8db,
    0x3fb9f4902d55d1f9,
    0x3c52696d7eac1dc1,
    0x3fefd5c94b43e000,
    0xbc62e768cb4f92f9,
    0x3fbbf1b78568391d,
    0x3c5e91841dea4cc8,
    0x3fefcf0c800e99b1,
    0x3c6ea3d786d186ac,
    0x3fbdee6f16c1cce6,
    0xbc450f8e2fb71673,
    0x3fefc7d078d1bc88,
    0x3c8075d2447db685,
    0x3fbfeaaeee86ee36,
    0xbc4afcb2bcc6f03b,
    0x3fefc015527d5bd3,
    0x3c8b68f35094efb8,
    0x3fc0f3378ddd71d1,
    0x3c6d8468724f0f9e,
    0x3fefb7db2bfe0695,
    0x3c821dadf4f65ab1,
    0x3fc1f0d3d7afceaf,
    0xbc66ef95099769a5,
    0x3fefaf22263c4bd3,
    0xbc552ace133a2769,
    0x3fc2ee285e4ab88f,
    0xbc6e4d0f05dee058,
    0x3fefa5ea641c36f2,
    0x3c404da6ed17cc7c,
    0x3fc3eb312c5d66cb,
    0x3c647d666b66cb91,
    0x3fef9c340a7cc428,
    0x3c8c5b6b063b7462,
    0x3fc4e7ea4dc5f27b,
    0x3c5949db2ac072fc,
    0x3fef91ff40374d01,
    0xbc67d03f4d3a9e4c,
    0x3fc5e44fcfa126f3,
    0xbc66f443063f89b6,
    0x3fef874c2e1eecf6,
    0xbc8c6514e1332b16,
    0x3fc6e05dc05a4d4c,
    0xbbd32c5c8b81c940,
    0x3fef7c1afeffde24,
    0xbc78f55bc47540b1,
    0x3fc7dc102fbaf2b5,
    0x3c45ab50e23c97c3,
    0x3fef706bdf9ece1c,
    0xbc8698c80c36dcb4,
    0x3fc8d7632efaa944,
    0xbc620fa262cbb953,
    0x3fef643efeb82acd,
    0x3c76b00ac1fe28ac,
    0x3fc9d252d0cec312,
    0x3c59c43d80b1137d,
    0x3fef57948cff6797,
    0x3c6e3a0d3e03b1d5,
    0x3fcaccdb297a0765,
    0xbc59883b57d6cdeb,
    0x3fef4a6cbd1e3a79,
    0x3c813df0edaebb57,
    0x3fcbc6f84edc6199,
    0x3c69c1a56a7b0cab,
    0x3fef3cc7c3b3d16e,
    0xbc621a3ad28a3494,
    0x3fccc0a6588289a3,
    0xbc6868d09bc87c6b,
    0x3fef2ea5d753ffed,
    0x3c8cc4215f56d583,
    0x3fcdb9e15fb5a5d0,
    0xbc632e20d6cc6fc2,
    0x3fef20073086649f,
    0x3c7b940416c1984b,
    0x3fceb2a57f8ae5a3,
    0xbc60be06af572ceb,
    0x3fef10ec09c5873b,
    0x3c8d9072762c1283,
    0x3fcfaaeed4f31577,
    0xbc615d88508e32b8,
    0x3fef01549f7deea1,
    0x3c8d3c1e99e5cafd,
    0x3fd0515cbf65155c,
    0xbc79b8c29dfd8ec8,
    0x3feef141300d2f26,
    0xbc82aa1b08ded372,
    0x3fd0cd00cef36436,
    0xbc79fb0a0c93e2b5,
    0x3feee0b1fbc0f11c,
    0xbc4bfd2380bbc3b1,
    0x3fd14861aa94ddeb,
    0xbc6be881b5b615a4,
    0x3feecfa744d5efa1,
    0xbc556d0a4af541d0,
    0x3fd1c37d64c6b876,
    0x3c746076fe0dcff5,
    0x3feebe214f76efa8,
    0xbc802f9f12ba543e,
    0x3fd23e52111aaf36,
    0xbc74f080334eff18,
    0x3feeac2061bbaf4f,
    0x3c62c1d53e94658d,
    0x3fd2b8ddc43eb49f,
    0x3c61553899f2d807,
    0x3fee99a4c3a7cd83,
    0xbc82264b1bc53ce8,
    0x3fd3331e94049f87,
    0x3c7e0cb6b40c302c,
    0x3fee86aebf29a9ed,
    0x3c89397afdbb58a7,
    0x3fd3ad129769d3d8,
    0x3c003d5504878398,
    0x3fee733ea0193d40,
    0xbc86428b3546ce13,
    0x3fd426b7e69ee697,
    0xbc7f09c75705c59f,
    0x3fee5f54b436e9d0,
    0x3c87eb0fd02fc8bc,
    0x3fd4a00c9b0f3d20,
    0x3c7823ba6bb08ead,
    0x3fee4af14b2a449c,
    0xbc868ca02e8a6833,
    0x3fd5190ecf68a77a,
    0x3c7b357155eef0f3,
    0x3fee3614b680d6a5,
    0xbc727793aa015237,
    0x3fd591bc9fa2f597,
    0x3c67c74bac3fe0cb,
    0x3fee20bf49acd6c1,
    0xbc5660aec7ef636c,
    0x3fd60a1429078775,
    0x3c5b1fd80ba89133,
    0x3fee0af15a03dbce,
    0x3c5fe8e702771ae6,
    0x3fd682138a38d7f7,
    0xbc7d889202444aad,
    0x3fedf4ab3ebd875e,
    0xbc8e2d8a7e6736c4,
    0x3fd6f9b8e33a0255,
    0x3c742bc14ee9da0d,
    0x3feddded50f228d6,
    0xbc6e80c8d42ba2bf,
    0x3fd7710255764214,
    0xbc66ead7314bb6ce,
    0x3fedc6b7eb995912,
    0x3c54b364776dcd35,
    0x3fd7e7ee03c86d4e,
    0xbc7b63bcdabf5af2,
    0x3fedaf0b6b888e83,
    0x3c8a249e2b5e5cea,
    0x3fd85e7a12826949,
    0x3c78a40e9b5face0,
    0x3fed96e82f71a9dc,
    0x3c8ff61bd5d2039d,
    0x3fd8d4a4a774992f,
    0x3c744a02ea766326,
    0x3fed7e4e97e17b4a,
    0xbc63b770352bed94,
    0x3fd94a6be9f546c5,
    0xbc769ce13e683f58,
    0x3fed653f073e4040,
    0xbc876236434bec37,
    0x3fd9bfce02e80510,
    0x3c709e39a320b0a4,
    0x3fed4bb9e1c619e0,
    0x3c8f34bb77858f61,
    0x3fda34c91cc50cca,
    0xbc5a310e3b50cecd,
    0x3fed31bf8d8d7c06,
    0x3c7e60dd3089cbdd,
    0x3fdaa95b63a09277,
    0xbc66293eb13c0381,
    0x3fed1750727d94f0,
    0x3c80d52b1ec1a48e,
    0x3fdb1d8305321617,
    0xbc7ae242cb99f519,
    0x3fecfc6cfa52ad9f,
    0x3c88b5b5508f2a0d,
    0x3fdb913e30dbac43,
    0xbc7e38ad2f6c3ff1,
    0x3fece115909a82e5,
    0x3c81f139bb31109a,
    0x3fdc048b17b140a3,
    0x3c619fe6757e9fa7,
    0x3fecc54aa2b2972e,
    0x3c64ee162ba83a98,
    0x3fdc7767ec7fd19e,
    0xbc5eb14d1a3d5826,
    0x3feca90c9fc67d0b,
    0xbc646a81485e3462,
    0x3fdce9d2e3d4a51f,
    0xbc62fc8a12dae298,
    0x3fec8c5bf8ce1a84,
    0x3c7ab3d1a1590123,
    0x3fdd5bca34047661,
    0x3c728a44a75fc29c,
    0x3fec6f39208be53b,
    0xbc8741dbfbaadb42,
    0x3fddcd4c15329c9a,
    0x3c70d4c6e171fd9a,
    0x3fec51a48b8b175e,
    0xbc61bbb43b9aa880,
    0x3fde3e56c1582a69,
    0xbc50a4821099f88f,
    0x3fec339eb01ddd81,
    0xbc8caaf5ee82c5c0,
    0x3fdeaee8744b05f0,
    0xbc5789b43c9b027d,
    0x3fec1528065b7d50,
    0xbc8892111312e828,
    0x3fdf1eff6bc4f97b,
    0x3c717212f8a7525c,
    0x3febf641081e7536,
    0x3c8b7bd71628a9a1,
    0x3fdf8e99e76abc97,
    0x3c59d950af2d00a3,
    0x3febd6ea310294f5,
    0x3c731bbcc88c109d,
    0x3fdffdb628d2f57a,
    0x3c6f4a992e905b6a,
    0x3febb723fe630f32,
    0x3c772bd2452d0a39,
    0x3fe0362939c69955,
    0xbc82d8cd78397b01,
    0x3feb96eeef58840e,
    0x3c545a3cc78fade0,
    0x3fe06d3686946e5b,
    0x3c83f5ae4538ff1b,
    0x3feb764b84b704c2,
    0xbc8f5848c21b389b,
    0x3fe0a4021e9e1001,
    0xbc86f643a13914f6,
    0x3feb553a410c104e,
    0x3c58ff7947027a16,
    0x3fe0da8b26b5672e,
    0xbc8a58def0bee909,
    0x3feb33bba89c8948,
    0x3c8ea6a51d1f6ca9,
    0x3fe110d0c4b69c3b,
    0x3c8d918998809981,
    0x3feb11d04162a4c6,
    0x3c71dd561efbc0c2,
    0x3fe146d21f8b7f82,
    0x3c7bf9535e2739a8,
    0x3feaef78930bd275,
    0xbc7f836279746f94,
    0x3fe17c8e5f2eedb0,
    0x3c635e57102e2488,
    0x3feaccb526f69de5,
    0x3c88fb6a8dd6b6cc,
    0x3fe1b204acb02fdd,
    0xbc5f190c70cbb5ff,
    0x3feaa98688308913,
    0xbc0b83d607cd5070,
    0x3fe1e7343236574c,
    0x3c722a3fa4f41d5a,
    0x3fea85ed4373e02d,
    0x3c69be06385ec792,
    0x3fe21c1c1b0394cf,
    0x3c5e5b324b23aa31,
    0x3fea61e9e72586af,
    0x3c858330e2fd453f,
    0x3fe250bb93788bbb,
    0x3c7ea3d02457bcce,
    0x3fea3d7d0352bdcf,
    0xbc868dbaeca19669,
    0x3fe28511c917a067,
    0xbc801df1d9a16b70,
    0x3fea18a729aee445,
    0x3c395e25736c0358,
    0x3fe2b91dea88421e,
    0xbc8fa371db216ab0,
    0x3fe9f368ed912f85,
    0xbc81d200c5791606,
    0x3fe2ecdf279a3082,
    0x3c8d3557e0e7e37e,
    0x3fe9cdc2e3f25e5c,
    0x3c83f99112993f62,
    0x3fe32054b148bc4f,
    0x3c8f6b42095a135b,
    0x3fe9a7b5a36a6514,
    0x3c8722cfcc9fa7a9,
    0x3fe3537db9be0367,
    0x3c6b327e7af040f0,
    0x3fe98141c42e1310,
    0x3c8d1ff80488f08d,
    0x3fe386597456282b,
    0xbc710fada93b07a8,
    0x3fe95a67e00cb1fd,
    0xbc80befda21f862d,
    0x3fe3b8e715a2840a,
    0xbc797653a7d2f07b,
    0x3fe93328926d9e92,
    0xbc8bb77003600cda,
    0x3fe3eb25d36cd53a,
    0xbc5be570e1570fc0,
    0x3fe90b84784ddaf7,
    0xbc70feb10ab93b87,
    0x3fe41d14e4ba6790,
    0x3c84608fd287ecf5,
    0x3fe8e37c303d9ad1,
    0xbc6463a4b53d4bf8,
    0x3fe44eb381cf386b,
    0xbc83ed6c1e6a5505,
    0x3fe8bb105a5dc900,
    0x3c8863e03e9474c1,
    0x3fe48000e431159f,
    0xbc8b194a7463ed10,
    0x3fe89241985d871f,
    0x3c8c48d9c413ed84,
    0x3fe4b0fc46aab761,
    0x3c20da05738cc59a,
    0x3fe869108d77a6c6,
    0x3c7338ffe2bfe9dd,
    0x3fe4e1a4e54ed51b,
    0xbc8a492f89b7c76a,
    0x3fe83f7dde701ca0,
    0xbc4152cf609bc6e8,
    0x3fe511f9fd7b351c,
    0xbc85c0e861c48831,
    0x3fe8158a31916d5d,
    0xbc6de8b90b8228de,
    0x3fe541facddbb724,
    0x3c7232c28520d391,
    0x3fe7eb362eaa1488,
    0x3c5a1d65a4a5959f,
    0x3fe571a6966d59b3,
    0x3c5c843b4d0fb198,
    0x3fe7c0827f09e54f,
    0xbc6c73d6d72aee68,
    0x3fe5a0fc98813a12,
    0xbc8d82e2b7d4227b,
    0x3fe7956fcd7f6543,
    0xbc8ab276e9d45ae4,
    0x3fe5cffc16bf8f0d,
    0x3c896cb370eb578a,
    0x3fe769fec655211f,
    0xbc6827d5cf8c68c5,
    0x3fe5fea4552a9e57,
    0x3c80b6cef7ee20b7,
    0x3fe73e30174efba1,
    0xbc65d3ae3d94ad5f,
    0x3fe62cf49921ac79,
    0xbc8edd9855b6241a,
    0x3fe712046fa77678,
    0x3c8425b0a5029c81,
    0x3fe65aec2963e755,
    0x3c8126f96b71053c,
    0x3fe6e57c800cf55e,
    0x3c860286dedbd0a6,
    0x3fe6888a4e134b2f,
    0xbc86b7d37644d5e6,
    0x3fe6b898fa9efb5d,
    0x3c715ac786ccf4b2,
    0x3fe6b5ce50b7821a,
    0xbc65d5158f702e0f,
    0x3fe68b5a92eb6253,
    0xbc89a91ad985f89c,
    0x3fe6e2b77c40bde1,
    0xbc70e729857fad53,
    0x3fe65dc1fdeb8cba,
    0xbc597c1b47337c77,
    0x3fe70f451d0a8c40,
    0x3c697ede3885770d,
    0x3fe62fcff20191c7,
    0x3c6d9143895756ef,
    0x3fe73b7680dea578,
    0xbc72248306dc12a2,
    0x3fe6018526f563df,
    0x3c846ca5e0e432d0,
    0x3fe7674af6f7b524,
    0x3c7e9d3f94ac84a8,
    0x3fe5d2e255f1f17a,
    0x3c80314104c8892b,
    0x3fe792c1d0041d52,
    0xbc8abf05eeb354eb,
    0x3fe5a3e839824077,
    0x3c8428aa2759be62,
    0x3fe7bdda5e28b3c2,
    0x3c4ad1197ccd0393,
    0x3fe574978d8e83f2,
    0x3c8f4714af282d23,
    0x3fe7e893f5037959,
    0x3c80eefbaa650c4c,
    0x3fe544f10f592ca5,
    0xbc8e7ae8e6c7a62f,
    0x3fe812ede9ae4ba4,
    0xbc87830adf402dda,
    0x3fe514f57d7bf3da,
    0x3c747a108073c259,
];

#[inline]
fn tab(k: usize) -> f64 {
    f64::from_bits(SINCOSTAB[k])
}

/// `u.i[LOW_HALF] << 2` of `u = big + |x|`: the table row.
#[inline]
fn row(u: f64) -> usize {
    ((u.to_bits() as u32 as i32) << 2) as usize
}

/// `do_cos(x, dx)`.
fn do_cos(x: f64, dx: f64) -> f64 {
    let dx = if x < 0.0 { -dx } else { dx };
    let u = BIG + x.abs();
    let x = x.abs() - (u - BIG) + dx;
    let xx = x * x;
    let s = fma(x * xx, fma(xx, SN5, SN3), x);
    let c = xx * fma(xx, fma(xx, CS6, CS4), CS2);
    let k = row(u);
    let (sn, ssn, cs, ccs) = (tab(k), tab(k + 1), tab(k + 2), tab(k + 3));
    let cor = fma(-sn, s, fma(-cs, c, fma(-s, ssn, ccs)));
    cs + cor
}

/// `TAYLOR_SIN(xx, x, dx)`.
fn taylor_sin(xx: f64, x: f64, dx: f64) -> f64 {
    let poly = fma(fma(fma(fma(S5, xx, S4), xx, S3), xx, S2), xx, S1);
    let t = fma(fma(poly, x, -(0.5 * dx)), xx, dx);
    x + t
}

/// `do_sin(x, dx)`.
fn do_sin(x: f64, dx: f64) -> f64 {
    let xold = x;
    if x.abs() < 0.126 {
        return taylor_sin(x * x, x, dx);
    }
    let dx = if x <= 0.0 { -dx } else { dx };
    let u = BIG + x.abs();
    let x = x.abs() - (u - BIG);
    let xx = x * x;
    let s = x + fma(x * xx, fma(xx, SN5, SN3), dx);
    let c = fma(x, dx, xx * fma(xx, fma(xx, CS6, CS4), CS2));
    let k = row(u);
    let (sn, ssn, cs, ccs) = (tab(k), tab(k + 1), tab(k + 2), tab(k + 3));
    let cor = fma(cs, s, fma(-sn, c, fma(s, ccs, ssn)));
    (sn + cor).copysign(xold)
}

/// `reduce_sincos(x)` → `(n, a, da)`.
fn reduce_sincos(x: f64) -> (i32, f64, f64) {
    let t = fma(x, HPINV, TOINT);
    let xn = t - TOINT;
    let y = fma(-xn, MP2, fma(-xn, MP1, x));
    let n = (t.to_bits() as u32 & 3) as i32;
    let t2 = fma(-xn, PP3, y);
    let mut db = fma(-xn, PP3, y - t2);
    let b = fma(-xn, PP4, t2);
    db += fma(-xn, PP4, t2 - b);
    (n, b, db)
}

fn do_sincos(a: f64, da: f64, n: i32) -> f64 {
    let r = if n & 1 != 0 {
        do_cos(a, da)
    } else {
        do_sin(a, da)
    };
    if n & 2 != 0 { -r } else { r }
}

/// `__branred(x)` → `(n, a, aa)` (no FMA).
fn branred(x: f64) -> (i32, f64, f64) {
    const T576: f64 = f64::from_bits(0x63f0000000000000);
    const TM600: f64 = f64::from_bits(0x1a70000000000000);
    const TM24: f64 = f64::from_bits(0x3e70000000000000);
    const BIGB: f64 = f64::from_bits(0x4338000000000000);
    const BIG1: f64 = f64::from_bits(0x4358000000000000);
    const BHP0: f64 = f64::from_bits(0x3FF921FB54442D18);
    const BHP1: f64 = f64::from_bits(0x3C91A62633145C07);
    const BMP1: f64 = f64::from_bits(0x3FF921FB58000000);
    const BMP2: f64 = f64::from_bits(0xBE4DDE9740000000);
    const SPLIT: f64 = 134217729.0;
    let x = x * TM600;
    let t = x * SPLIT;
    let x1 = t - (t - x);
    let x2 = x - x1;
    let half = |xp: f64| -> (f64, f64, f64) {
        let mut sum = 0.0f64;
        let mut k = (((xp.to_bits() >> 32) as u32 >> 20) & 2047) as i32;
        k = (k - 450) / 24;
        if k < 0 {
            k = 0;
        }
        let mut gor = f64::from_bits(T576.to_bits() - (((k * 24) as u64) << 52));
        let mut r = [0.0f64; 6];
        for (i, ri) in r.iter_mut().enumerate() {
            *ri = xp * TOVERP[k as usize + i] * gor;
            gor *= TM24;
        }
        for ri in r.iter_mut().take(3) {
            let s = (*ri + BIGB) - BIGB;
            sum += s;
            *ri -= s;
        }
        let mut t = 0.0f64;
        for i in 0..6 {
            t += r[5 - i];
        }
        let mut bb = (((((r[0] - t) + r[1]) + r[2]) + r[3]) + r[4]) + r[5];
        let s = (t + BIGB) - BIGB;
        sum += s;
        let t = t - s;
        let b = t + bb;
        bb = (t - b) + bb;
        let s = (sum + BIG1) - BIG1;
        sum -= s;
        (b, bb, sum)
    };
    let (b1, bb1, sum1) = half(x1);
    let (b2, bb2, sum2) = half(x2);
    let mut sum = sum1 + sum2;
    let mut b = b1 + b2;
    let bb = if b1.abs() > b2.abs() {
        (b1 - b) + b2
    } else {
        (b2 - b) + b1
    };
    if b > 0.5 {
        b -= 1.0;
        sum += 1.0;
    } else if b < -0.5 {
        b += 1.0;
        sum -= 1.0;
    }
    let s = b + (bb + bb1 + bb2);
    let t = ((b - s) + bb) + (bb1 + bb2);
    let bs = s * SPLIT;
    let t1 = bs - (bs - s);
    let t2 = s - t1;
    let b = s * BHP0;
    let bb = (((t1 * BMP1 - b) + t1 * BMP2) + t2 * BMP1) + (t2 * BMP2 + s * BHP1 + t * BHP0);
    let s = b + bb;
    let t = (b - s) + bb;
    ((sum as i32) & 3, s, t)
}

/// glibc `sin(x)`.
pub fn sin(x: f64) -> f64 {
    let k = ((x.to_bits() >> 32) as u32) & 0x7fff_ffff;
    if k < 0x3e50_0000 {
        x
    } else if k < 0x3feb_6000 {
        do_sin(x, 0.0)
    } else if k < 0x4003_68fd {
        let t = HP0 - x.abs();
        do_cos(t, HP1).copysign(x)
    } else if k < 0x4199_21FB {
        let (n, a, da) = reduce_sincos(x);
        do_sincos(a, da, n)
    } else if k < 0x7ff0_0000 {
        let (n, a, da) = branred(x);
        do_sincos(a, da, n)
    } else {
        x / x
    }
}

/// glibc `cos(x)`.
pub fn cos(x: f64) -> f64 {
    let k = ((x.to_bits() >> 32) as u32) & 0x7fff_ffff;
    if k < 0x3e40_0000 {
        1.0
    } else if k < 0x3feb_6000 {
        do_cos(x, 0.0)
    } else if k < 0x4003_68fd {
        let y = HP0 - x.abs();
        let a = y + HP1;
        let da = (y - a) + HP1;
        do_sin(a, da)
    } else if k < 0x4199_21FB {
        let (n, a, da) = reduce_sincos(x);
        do_sincos(a, da, n + 1)
    } else if k < 0x7ff0_0000 {
        let (n, a, da) = branred(x);
        do_sincos(a, da, n + 1)
    } else {
        x / x
    }
}

// Generated from glibc 2.43 sysdeps/ieee754/dbl-64/utan.h, utan.tbl (IBM Accurate Mathematical Library) by script.

const T_D3: f64 = f64::from_bits(0x3fd5555555555555);
const T_D5: f64 = f64::from_bits(0x3fc11111111107c6);
const T_D7: f64 = f64::from_bits(0x3faba1ba1cdb8745);
const T_D9: f64 = f64::from_bits(0x3f9664ed49cfc666);
const T_D11: f64 = f64::from_bits(0x3f82385a3cf2e4ea);
const T_E0: f64 = f64::from_bits(0x3fd5555555554dbd);
const T_E1: f64 = f64::from_bits(0x3fc11112e0a6b45f);
const T_MFFTNHF: f64 = f64::from_bits(0xc02f000000000000);
const T_G1: f64 = f64::from_bits(0x3e4b096c00000000);
const T_G2: f64 = f64::from_bits(0x3faf212d00000000);
const T_G3: f64 = f64::from_bits(0x3fe92f1a00000000);
const T_G4: f64 = f64::from_bits(0x4039000000000000);
const T_G5: f64 = f64::from_bits(0x4197d78400000000);
const T_GY2: f64 = f64::from_bits(0x3faf212d00000000);
const T_MP1: f64 = f64::from_bits(0x3ff921fb58000000);
const T_MP2: f64 = f64::from_bits(0xbe4dde973c000000);
const T_MP3: f64 = f64::from_bits(0xbc8cb3b399d747f2);
const T_PP3: f64 = f64::from_bits(0xbc8cb3b398000000);
const T_PP4: f64 = f64::from_bits(0xbacd747f23e32ed7);
const T_HPINV: f64 = f64::from_bits(0x3fe45f306dc9c883);
const T_TOINT: f64 = f64::from_bits(0x4338000000000000);

/// `xfg[i] = (xi, Fi, Gi, FFi)`, `i = 16..=201`.
const XFG: [[f64; 4]; 186] = [
    [
        f64::from_bits(0x3fb000001e519d60),
        f64::from_bits(0x3fb0055796c4e240),
        f64::from_bits(0x402ff554628127b7),
        f64::from_bits(0xbb9a1dee9e355b06),
    ],
    [
        f64::from_bits(0x3fb100001b1a7010),
        f64::from_bits(0x3fb10668aab892b7),
        f64::from_bits(0x402e12c7be3fdf74),
        f64::from_bits(0x3ba89234037da741),
    ],
    [
        f64::from_bits(0x3fb200002505e350),
        f64::from_bits(0x3fb2079bff547824),
        f64::from_bits(0x402c65c5de853633),
        f64::from_bits(0x3bb7486ee9614250),
    ],
    [
        f64::from_bits(0x3fb2fffffcdc4252),
        f64::from_bits(0x3fb308f35eb16c68),
        f64::from_bits(0x402ae5dae56be74f),
        f64::from_bits(0xbb82c72691a23034),
    ],
    [
        f64::from_bits(0x3fb3ffffe3ff849f),
        f64::from_bits(0x3fb40a71154999cc),
        f64::from_bits(0x40298c43046b7352),
        f64::from_bits(0x3b9aceaf3843738f),
    ],
    [
        f64::from_bits(0x3fb4ffffedc9590f),
        f64::from_bits(0x3fb50c17429bdd80),
        f64::from_bits(0x4028538491b5d674),
        f64::from_bits(0xbbc1d02db4403d22),
    ],
    [
        f64::from_bits(0x3fb6000000ee83f7),
        f64::from_bits(0x3fb60de7da80cc21),
        f64::from_bits(0x40273724ef21a2a7),
        f64::from_bits(0xbb95e53c72523ffd),
    ],
    [
        f64::from_bits(0x3fb6ffffeb05ea41),
        f64::from_bits(0x3fb70fe4b8c51bea),
        f64::from_bits(0x40263370fae562ff),
        f64::from_bits(0xbb99ad0e8ffe0626),
    ],
    [
        f64::from_bits(0x3fb7ffffdc0515f7),
        f64::from_bits(0x3fb812101db54498),
        f64::from_bits(0x402545530e7eab5c),
        f64::from_bits(0xbb914c87d62ed686),
    ],
    [
        f64::from_bits(0x3fb8ffffe384d7ab),
        f64::from_bits(0x3fb9146c2a8d3727),
        f64::from_bits(0x40246a33fd57f3fd),
        f64::from_bits(0xbbbbda8d5381e06d),
    ],
    [
        f64::from_bits(0x3fb9ffffe4832347),
        f64::from_bits(0x3fba16fad50e1050),
        f64::from_bits(0x40239fe2c5537a96),
        f64::from_bits(0x3bc7f695c111eabb),
    ],
    [
        f64::from_bits(0x3fbb0000274540e3),
        f64::from_bits(0x3fbb19be7ae68517),
        f64::from_bits(0x4022e4813637e946),
        f64::from_bits(0x3bc307f88dbd9d93),
    ],
    [
        f64::from_bits(0x3fbbfffffebf2e9b),
        f64::from_bits(0x3fbc1cb88369cd19),
        f64::from_bits(0x4022367617aef223),
        f64::from_bits(0x3bc50038424a9cf3),
    ],
    [
        f64::from_bits(0x3fbd000023529045),
        f64::from_bits(0x3fbd1febc11d7ef7),
        f64::from_bits(0x4021945fb8e43d4e),
        f64::from_bits(0x3b81200752a6f224),
    ],
    [
        f64::from_bits(0x3fbdffffd872a829),
        f64::from_bits(0x3fbe23598ee4d6b7),
        f64::from_bits(0x4020fd0c76195d5f),
        f64::from_bits(0xbbb4d9ab85fdca85),
    ],
    [
        f64::from_bits(0x3fbeffffff323b84),
        f64::from_bits(0x3fbf2704ec9073e5),
        f64::from_bits(0x40206f713020200f),
        f64::from_bits(0x3bb77aa212836992),
    ],
    [
        f64::from_bits(0x3fc000000ce79195),
        f64::from_bits(0x3fc01577bc30cc61),
        f64::from_bits(0x401fd549d6564a88),
        f64::from_bits(0xbbc8926f965c0ad0),
    ],
    [
        f64::from_bits(0x3fc07fffee40e918),
        f64::from_bits(0x3fc0978d8279ac01),
        f64::from_bits(0x401edbb59294bc03),
        f64::from_bits(0xbb80a5334aae45d6),
    ],
    [
        f64::from_bits(0x3fc100000cc091fd),
        f64::from_bits(0x3fc119c544dfb2f7),
        f64::from_bits(0x401df0bb067d8e18),
        f64::from_bits(0xbbcc2c184ff642a4),
    ],
    [
        f64::from_bits(0x3fc180000d9936a1),
        f64::from_bits(0x3fc19c1fb9085a4b),
        f64::from_bits(0x401d131a71ce3629),
        f64::from_bits(0xbbc365530669355b),
    ],
    [
        f64::from_bits(0x3fc1ffffed5f3188),
        f64::from_bits(0x3fc21e9dee74bf2d),
        f64::from_bits(0x401c41b6ff0cd655),
        f64::from_bits(0x3b8867f5478ecfc5),
    ],
    [
        f64::from_bits(0x3fc2800005f06a51),
        f64::from_bits(0x3fc2a141550b313f),
        f64::from_bits(0x401b7b921702e6d2),
        f64::from_bits(0xbbadab51380131fe),
    ],
    [
        f64::from_bits(0x3fc2fffffe3d339e),
        f64::from_bits(0x3fc3240aa75f76df),
        f64::from_bits(0x401abfc8fcb6409d),
        f64::from_bits(0x3bc60bcf0d291d83),
    ],
    [
        f64::from_bits(0x3fc37fffed888d6f),
        f64::from_bits(0x3fc3a6fb13cc5db7),
        f64::from_bits(0x401a0d8f8ed5320d),
        f64::from_bits(0x3bb8a48e4eef03ab),
    ],
    [
        f64::from_bits(0x3fc4000002ca050d),
        f64::from_bits(0x3fc42a13e25776bb),
        f64::from_bits(0x4019642dfa84c2bc),
        f64::from_bits(0xbbd0bd5dcc56516f),
    ],
    [
        f64::from_bits(0x3fc47ffff2531f5c),
        f64::from_bits(0x3fc4ad55deb73404),
        f64::from_bits(0x4018c2fef86e9035),
        f64::from_bits(0x3b9cffe75aa287c8),
    ],
    [
        f64::from_bits(0x3fc5000013774992),
        f64::from_bits(0x3fc530c27d0ee307),
        f64::from_bits(0x4018296c370caf35),
        f64::from_bits(0xbbcf75d1f91d6532),
    ],
    [
        f64::from_bits(0x3fc57fffedddcb2d),
        f64::from_bits(0x3fc5b45a5db4347d),
        f64::from_bits(0x401796ee52190c0e),
        f64::from_bits(0x3b88a25f17d5d076),
    ],
    [
        f64::from_bits(0x3fc5fffff41949a0),
        f64::from_bits(0x3fc6381f13bf986a),
        f64::from_bits(0x40170b092d2255fd),
        f64::from_bits(0xbb9bfb23b1bcd5e7),
    ],
    [
        f64::from_bits(0x3fc67ffff834d3a1),
        f64::from_bits(0x3fc6bc118ec85952),
        f64::from_bits(0x4016854c62cf2268),
        f64::from_bits(0x3b9ee53b82e39e04),
    ],
    [
        f64::from_bits(0x3fc6fffffd9106ea),
        f64::from_bits(0x3fc74032f298f6f7),
        f64::from_bits(0x401605511f4f84a9),
        f64::from_bits(0xbbb59c4a112634b8),
    ],
    [
        f64::from_bits(0x3fc780000f649a4f),
        f64::from_bits(0x3fc7c4846ca53abc),
        f64::from_bits(0x40158ab94809d175),
        f64::from_bits(0x3bc91c7573d3cd2e),
    ],
    [
        f64::from_bits(0x3fc7ffffef06bbd8),
        f64::from_bits(0x3fc84906df7d76ad),
        f64::from_bits(0x4015152edd2b30a6),
        f64::from_bits(0xbbbfa2da084c3eef),
    ],
    [
        f64::from_bits(0x3fc88000021c6334),
        f64::from_bits(0x3fc8cdbbd965f986),
        f64::from_bits(0x4014a46251b74296),
        f64::from_bits(0xbb9ec02e74dcfe0b),
    ],
    [
        f64::from_bits(0x3fc8fffff38d0756),
        f64::from_bits(0x3fc952a428e173c7),
        f64::from_bits(0x4014380b17b59ebd),
        f64::from_bits(0xbbcd0f1cb77589f0),
    ],
    [
        f64::from_bits(0x3fc98000104efca1),
        f64::from_bits(0x3fc9d7c14644d23c),
        f64::from_bits(0x4013cfe5cb1eabd5),
        f64::from_bits(0xbbd5d6f7ea188d9e),
    ],
    [
        f64::from_bits(0x3fca000009417b30),
        f64::from_bits(0x3fca5d14096d76aa),
        f64::from_bits(0x40136bb4b3723db0),
        f64::from_bits(0x3bbe3e0dfbf3979c),
    ],
    [
        f64::from_bits(0x3fca7fffeb1c23ec),
        f64::from_bits(0x3fcae29dab60288d),
        f64::from_bits(0x40130b3e783071d7),
        f64::from_bits(0xbbc7dd823d5384bf),
    ],
    [
        f64::from_bits(0x3fcafffffb171c13),
        f64::from_bits(0x3fcb685fa221a96b),
        f64::from_bits(0x4012ae4dd8c0747d),
        f64::from_bits(0x3bd4644bd5554972),
    ],
    [
        f64::from_bits(0x3fcb80000aba44be),
        f64::from_bits(0x3fcbee5aecdf241f),
        f64::from_bits(0x401254b1c6fad63b),
        f64::from_bits(0x3ba41916d092b85a),
    ],
    [
        f64::from_bits(0x3fcc0000113d2a3e),
        f64::from_bits(0x3fcc7490b3e92543),
        f64::from_bits(0x4011fe3c9a62c035),
        f64::from_bits(0xbba3cc3941a03739),
    ],
    [
        f64::from_bits(0x3fcc7ffff49e00ce),
        f64::from_bits(0x3fccfb020f59eab0),
        f64::from_bits(0x4011aac3e956a631),
        f64::from_bits(0xbbb7a383bfa8cb5b),
    ],
    [
        f64::from_bits(0x3fcd000005f611ab),
        f64::from_bits(0x3fcd81b089e6844e),
        f64::from_bits(0x40115a1ff391268d),
        f64::from_bits(0x3bd39b5cb2dc91f3),
    ],
    [
        f64::from_bits(0x3fcd800014764ceb),
        f64::from_bits(0x3fce089d27debf0d),
        f64::from_bits(0x40110c2bfbc84740),
        f64::from_bits(0x3bc14d4d84712510),
    ],
    [
        f64::from_bits(0x3fce000014bcea76),
        f64::from_bits(0x3fce8fc916dbc820),
        f64::from_bits(0x4010c0c5a00ca48e),
        f64::from_bits(0xbbd33788640f1b9e),
    ],
    [
        f64::from_bits(0x3fce7ffffd7995bd),
        f64::from_bits(0x3fcf173588b50424),
        f64::from_bits(0x401077ccbe02169a),
        f64::from_bits(0xbbb61fee221fdf77),
    ],
    [
        f64::from_bits(0x3fcf00000cc35436),
        f64::from_bits(0x3fcf9ee3fd21a40b),
        f64::from_bits(0x401031231ee7ffe8),
        f64::from_bits(0x3bd427e3c79ff5c1),
    ],
    [
        f64::from_bits(0x3fcf800001d1da33),
        f64::from_bits(0x3fd0136ab7dbe15c),
        f64::from_bits(0x400fd95977d559e5),
        f64::from_bits(0x3bb0c6a1d67948d7),
    ],
    [
        f64::from_bits(0x3fd00000060c13b2),
        f64::from_bits(0x3fd05785aaad4f18),
        f64::from_bits(0x400f549e2675d182),
        f64::from_bits(0xbbc1520818f0dd10),
    ],
    [
        f64::from_bits(0x3fd0400003885492),
        f64::from_bits(0x3fd09bc3660542d7),
        f64::from_bits(0x400ed3e2df3f5fec),
        f64::from_bits(0xbbd95657b883ae62),
    ],
    [
        f64::from_bits(0x3fd08000052f5a13),
        f64::from_bits(0x3fd0e0249a195045),
        f64::from_bits(0x400e56f8fa68f2c8),
        f64::from_bits(0x3bded7ba5a543e8e),
    ],
    [
        f64::from_bits(0x3fd0c00002ba1af5),
        f64::from_bits(0x3fd124a9e2e7f24b),
        f64::from_bits(0x400dddb4bffe633f),
        f64::from_bits(0xbbdcba860c60278f),
    ],
    [
        f64::from_bits(0x3fd0fffff76642c1),
        f64::from_bits(0x3fd16953e162ffe6),
        f64::from_bits(0x400d67ed0311d5d5),
        f64::from_bits(0x3b7b1f4ae40c5f9e),
    ],
    [
        f64::from_bits(0x3fd14000033602f0),
        f64::from_bits(0x3fd1ae235f49508e),
        f64::from_bits(0x400cf57ab8708266),
        f64::from_bits(0xbbd6a6c28620f301),
    ],
    [
        f64::from_bits(0x3fd17ffffefd1a13),
        f64::from_bits(0x3fd1f318db2a9ba1),
        f64::from_bits(0x400c86398d11009e),
        f64::from_bits(0x3bd3a9c669b21d3b),
    ],
    [
        f64::from_bits(0x3fd1bffff718365d),
        f64::from_bits(0x3fd238350c41e3ac),
        f64::from_bits(0x400c1a06e02be47c),
        f64::from_bits(0x3bdb961a129e8cd1),
    ],
    [
        f64::from_bits(0x3fd1ffffff001e00),
        f64::from_bits(0x3fd27d78b2f6395e),
        f64::from_bits(0x400bb0c1f2fe9a85),
        f64::from_bits(0x3be074a9e68fd7d8),
    ],
    [
        f64::from_bits(0x3fd23ffffe425a6a),
        f64::from_bits(0x3fd2c2e4618faabe),
        f64::from_bits(0x400b4a4c190b18df),
        f64::from_bits(0xbbdf0d1ff615aad1),
    ],
    [
        f64::from_bits(0x3fd28000059ec1db),
        f64::from_bits(0x3fd30878d8583884),
        f64::from_bits(0x400ae6880cd82bc2),
        f64::from_bits(0xbbd563c3141c1f8d),
    ],
    [
        f64::from_bits(0x3fd2c000000dd081),
        f64::from_bits(0x3fd34e36affdb6d8),
        f64::from_bits(0x400a855a5270fc15),
        f64::from_bits(0xbbc6d88d9f2cdafd),
    ],
    [
        f64::from_bits(0x3fd2fffffc1dcd2b),
        f64::from_bits(0x3fd3941ea95875bc),
        f64::from_bits(0x400a26a8aa9502b6),
        f64::from_bits(0xbbe13cad8389b15c),
    ],
    [
        f64::from_bits(0x3fd33ffff6c0d4a0),
        f64::from_bits(0x3fd3da31739845f5),
        f64::from_bits(0x4009ca5a4d2573a0),
        f64::from_bits(0xbbc71636acaee379),
    ],
    [
        f64::from_bits(0x3fd3800006b16793),
        f64::from_bits(0x3fd4206fdbc088f0),
        f64::from_bits(0x400970579344e33a),
        f64::from_bits(0xbbc2c0521d7a4f81),
    ],
    [
        f64::from_bits(0x3fd3c00007358fa3),
        f64::from_bits(0x3fd466da6f23311d),
        f64::from_bits(0x4009188a5aa612ea),
        f64::from_bits(0x3b8653a5685e8edc),
    ],
    [
        f64::from_bits(0x3fd3fffffc3b18cf),
        f64::from_bits(0x3fd4ad71e9282e6b),
        f64::from_bits(0x4008c2dd641e643d),
        f64::from_bits(0x3b95f0ef3f567c64),
    ],
    [
        f64::from_bits(0x3fd44000000dd2a8),
        f64::from_bits(0x3fd4f4371fa3f2d1),
        f64::from_bits(0x40086f3c6072f821),
        f64::from_bits(0x3bb68efa95ff68b5),
    ],
    [
        f64::from_bits(0x3fd47ffffbb43713),
        f64::from_bits(0x3fd53b2ab3ac333c),
        f64::from_bits(0x40081d943da56692),
        f64::from_bits(0xbbbf4d7f2985fd3f),
    ],
    [
        f64::from_bits(0x3fd4bffffb113bf4),
        f64::from_bits(0x3fd5824d6e8ed9c2),
        f64::from_bits(0x4007cdd2a8add00f),
        f64::from_bits(0x3bcf478a1c9b3657),
    ],
    [
        f64::from_bits(0x3fd4fffff7f087c9),
        f64::from_bits(0x3fd5c9a007446496),
        f64::from_bits(0x40077fe6444588eb),
        f64::from_bits(0xbbc177dca4eabb0c),
    ],
    [
        f64::from_bits(0x3fd54000088b3814),
        f64::from_bits(0x3fd61123564125f9),
        f64::from_bits(0x400733be6281a765),
        f64::from_bits(0xbbc2c52cf57051c4),
    ],
    [
        f64::from_bits(0x3fd57ffff7d55966),
        f64::from_bits(0x3fd658d7e194a5d5),
        f64::from_bits(0x4006e94b73b47d1f),
        f64::from_bits(0x3bda2fcff9996dc6),
    ],
    [
        f64::from_bits(0x3fd5c00008bf2490),
        f64::from_bits(0x3fd6a0beb775b28d),
        f64::from_bits(0x4006a07e15b6ec28),
        f64::from_bits(0xbbe0ca90aa5285b8),
    ],
    [
        f64::from_bits(0x3fd6000009fa853f),
        f64::from_bits(0x3fd6e8d865a66cfd),
        f64::from_bits(0x400659481c701269),
        f64::from_bits(0x3bd9ea958591e13a),
    ],
    [
        f64::from_bits(0x3fd6400007595fca),
        f64::from_bits(0x3fd73125c0556a7c),
        f64::from_bits(0x4006139bbaae9d02),
        f64::from_bits(0x3bd88aff40152b83),
    ],
    [
        f64::from_bits(0x3fd68000031687da),
        f64::from_bits(0x3fd779a792e2cfd0),
        f64::from_bits(0x4005cf6bcae0882b),
        f64::from_bits(0xbbd8a4a29f439451),
    ],
    [
        f64::from_bits(0x3fd6bffff5c8cfe2),
        f64::from_bits(0x3fd7c25e9fb452ed),
        f64::from_bits(0x40058cabc561f1cd),
        f64::from_bits(0xbbe371a6f6a37d74),
    ],
    [
        f64::from_bits(0x3fd6fffff81df231),
        f64::from_bits(0x3fd80b4bcfb4dab5),
        f64::from_bits(0x40054b4f8d3ca5d3),
        f64::from_bits(0x3bcb4686679dc99f),
    ],
    [
        f64::from_bits(0x3fd73ffffa71385e),
        f64::from_bits(0x3fd8546fe007a9b6),
        f64::from_bits(0x40050b4bb3b22176),
        f64::from_bits(0xbbcd1540a5c73477),
    ],
    [
        f64::from_bits(0x3fd78000024a9c2b),
        f64::from_bits(0x3fd89dcba7fcf5cf),
        f64::from_bits(0x4004cc953159cbe1),
        f64::from_bits(0xbbdc25ead58a6ad0),
    ],
    [
        f64::from_bits(0x3fd7c00002eb62b8),
        f64::from_bits(0x3fd8e75fec0ba5cf),
        f64::from_bits(0x40048f218731eeea),
        f64::from_bits(0xbbc1cb73cc1adafb),
    ],
    [
        f64::from_bits(0x3fd80000054a52d1),
        f64::from_bits(0x3fd9312d8bb822e9),
        f64::from_bits(0x400452e69170a729),
        f64::from_bits(0xbbd8bb17eac002ee),
    ],
    [
        f64::from_bits(0x3fd83ffff93a00a3),
        f64::from_bits(0x3fd97b354bb9ad2a),
        f64::from_bits(0x400417daae924e7f),
        f64::from_bits(0x3bd4b8009a378cc7),
    ],
    [
        f64::from_bits(0x3fd87ffffbdc91c1),
        f64::from_bits(0x3fd9c5782771b601),
        f64::from_bits(0x4003ddf478855799),
        f64::from_bits(0x3bd9077da00445d9),
    ],
    [
        f64::from_bits(0x3fd8bffff6d215e6),
        f64::from_bits(0x3fda0ff6e0ea4a0b),
        f64::from_bits(0x4003a52b189a0989),
        f64::from_bits(0xbbda683189c0613d),
    ],
    [
        f64::from_bits(0x3fd9000002f734ef),
        f64::from_bits(0x3fda5ab2736bf579),
        f64::from_bits(0x40036d75e9244ca6),
        f64::from_bits(0x3be3a6d84b722377),
    ],
    [
        f64::from_bits(0x3fd9400004eef8b4),
        f64::from_bits(0x3fdaa5ab9fb6e3d0),
        f64::from_bits(0x400336ccc9089cb7),
        f64::from_bits(0x3b9f696322cc00bb),
    ],
    [
        f64::from_bits(0x3fd98000041ec76a),
        f64::from_bits(0x3fdaf0e35176c7e4),
        f64::from_bits(0x40030127cb0b9506),
        f64::from_bits(0x3bb1ffdb5385a849),
    ],
    [
        f64::from_bits(0x3fd9c00008044e47),
        f64::from_bits(0x3fdb3c5a77071224),
        f64::from_bits(0x4002cc7f50d75ec7),
        f64::from_bits(0xbbb0fade78effc8a),
    ],
    [
        f64::from_bits(0x3fda000001f8235b),
        f64::from_bits(0x3fdb8811e725782e),
        f64::from_bits(0x400298cc18fbfb37),
        f64::from_bits(0xbbe55ed33b50e71b),
    ],
    [
        f64::from_bits(0x3fda3ffffb8c6f08),
        f64::from_bits(0x3fdbd40a97b086f3),
        f64::from_bits(0x40026607154de04b),
        f64::from_bits(0xbbdec65e455faae3),
    ],
    [
        f64::from_bits(0x3fda7ffffb3d63e1),
        f64::from_bits(0x3fdc20457d9a3b8a),
        f64::from_bits(0x400234297e60bfbb),
        f64::from_bits(0x3be3001c154ebd33),
    ],
    [
        f64::from_bits(0x3fdabffff5f45c48),
        f64::from_bits(0x3fdc6cc37b8d45e6),
        f64::from_bits(0x4002032cdb1ace69),
        f64::from_bits(0xbbe5ebf83ed33616),
    ],
    [
        f64::from_bits(0x3fdb00000508b34c),
        f64::from_bits(0x3fdcb985a27e8d37),
        f64::from_bits(0x4001d30ad4459a2b),
        f64::from_bits(0xbbd01432ae61e2d1),
    ],
    [
        f64::from_bits(0x3fdb40000a84710c),
        f64::from_bits(0x3fdd068cc3e50155),
        f64::from_bits(0x4001a3bd775034dd),
        f64::from_bits(0xbbe80b1e58e0e228),
    ],
    [
        f64::from_bits(0x3fdb7ffff692e9d8),
        f64::from_bits(0x3fdd53d9c49d6627),
        f64::from_bits(0x4001753efe18066a),
        f64::from_bits(0xbbb004c8f760d33e),
    ],
    [
        f64::from_bits(0x3fdbc0000280f14d),
        f64::from_bits(0x3fdda16de4e81013),
        f64::from_bits(0x40014789a38ea052),
        f64::from_bits(0x3be848bc27c9c4ea),
    ],
    [
        f64::from_bits(0x3fdc0000001121d1),
        f64::from_bits(0x3fddef49eac018f0),
        f64::from_bits(0x40011a9820b8be0c),
        f64::from_bits(0xbbe1527ed0d6010e),
    ],
    [
        f64::from_bits(0x3fdc3ffffef662aa),
        f64::from_bits(0x3fde3d6eea0c7070),
        f64::from_bits(0x4000ee6532f46ccd),
        f64::from_bits(0x3be8d241189a000d),
    ],
    [
        f64::from_bits(0x3fdc800009845818),
        f64::from_bits(0x3fde8bddf36a8b1b),
        f64::from_bits(0x4000c2ebcac73476),
        f64::from_bits(0x3bd221f712bed284),
    ],
    [
        f64::from_bits(0x3fdcbffffb0493bf),
        f64::from_bits(0x3fdeda97e0c60d10),
        f64::from_bits(0x40009827251c7836),
        f64::from_bits(0xbbe0bd546eec41b7),
    ],
    [
        f64::from_bits(0x3fdcfffffd52961f),
        f64::from_bits(0x3fdf299defb3e44b),
        f64::from_bits(0x40006e1274e459f5),
        f64::from_bits(0xbbd93f77e969c82f),
    ],
    [
        f64::from_bits(0x3fdd3ffffe2319a4),
        f64::from_bits(0x3fdf78f117139490),
        f64::from_bits(0x400044a93e737e94),
        f64::from_bits(0xbb91e7cc49594b7a),
    ],
    [
        f64::from_bits(0x3fdd7ffffa4de596),
        f64::from_bits(0x3fdfc892638f49e8),
        f64::from_bits(0x40001be7231057a5),
        f64::from_bits(0x3bd482b0f5af9f5f),
    ],
    [
        f64::from_bits(0x3fddbffffe729a69),
        f64::from_bits(0x3fe00c417c6ab019),
        f64::from_bits(0x3fffe78fbf612660),
        f64::from_bits(0x3bea5cda00da681e),
    ],
    [
        f64::from_bits(0x3fde000009d66802),
        f64::from_bits(0x3fe03461f6b883cf),
        f64::from_bits(0x3fff988ebc05a87c),
        f64::from_bits(0xbbe06c33f2372669),
    ],
    [
        f64::from_bits(0x3fde3ffffb211657),
        f64::from_bits(0x3fe05cab191db8e8),
        f64::from_bits(0x3fff4ac37bcfe6be),
        f64::from_bits(0xbbd5d51f5ed8d35b),
    ],
    [
        f64::from_bits(0x3fde80000a3f068a),
        f64::from_bits(0x3fe0851d95fb54f0),
        f64::from_bits(0x3ffefe26144ca408),
        f64::from_bits(0xbbc7c894a2c169c5),
    ],
    [
        f64::from_bits(0x3fdec00001adb060),
        f64::from_bits(0x3fe0adb9dc7b54f9),
        f64::from_bits(0x3ffeb2af5ebe52a7),
        f64::from_bits(0x3bd4e740312c5ffd),
    ],
    [
        f64::from_bits(0x3fdeffffff5c0d01),
        f64::from_bits(0x3fe0d68092550a8d),
        f64::from_bits(0x3ffe68580d71fdf0),
        f64::from_bits(0x3bddd8a696b35499),
    ],
    [
        f64::from_bits(0x3fdf3ffff93d5fcc),
        f64::from_bits(0x3fe0ff7245cb4374),
        f64::from_bits(0x3ffe1f193cce5040),
        f64::from_bits(0xbbc9f0ec7c1efab4),
    ],
    [
        f64::from_bits(0x3fdf7ffffa0dd18f),
        f64::from_bits(0x3fe1288f944dd508),
        f64::from_bits(0x3ffdd6ec298b874d),
        f64::from_bits(0x3bea6ebd9642a0a6),
    ],
    [
        f64::from_bits(0x3fdfbffffd3a9f1a),
        f64::from_bits(0x3fe151d913750f3e),
        f64::from_bits(0x3ffd8fca5806a27e),
        f64::from_bits(0x3bda2a03fc65ac7a),
    ],
    [
        f64::from_bits(0x3fdffffffc481400),
        f64::from_bits(0x3fe17b4f598944ca),
        f64::from_bits(0x3ffd49ad82532170),
        f64::from_bits(0x3bc4412e3d236dc3),
    ],
    [
        f64::from_bits(0x3fe01fffff53786c),
        f64::from_bits(0x3fe1a4f307d83d47),
        f64::from_bits(0x3ffd048f851bffeb),
        f64::from_bits(0x3bd1589d29f81b14),
    ],
    [
        f64::from_bits(0x3fe03ffffee301b7),
        f64::from_bits(0x3fe1cec4b8a6a382),
        f64::from_bits(0x3ffcc06a7c519db6),
        f64::from_bits(0x3bd370e65b24d6b2),
    ],
    [
        f64::from_bits(0x3fe06000006e36bf),
        f64::from_bits(0x3fe1f8c5114eb8be),
        f64::from_bits(0x3ffc7d38a34d6786),
        f64::from_bits(0xbbea92de4b98c1d4),
    ],
    [
        f64::from_bits(0x3fe07ffffd60aa43),
        f64::from_bits(0x3fe222f4abeccecb),
        f64::from_bits(0x3ffc3af477342ac4),
        f64::from_bits(0xbbdd47f603a5c2c2),
    ],
    [
        f64::from_bits(0x3fe0a000037762e8),
        f64::from_bits(0x3fe24d543f99efe8),
        f64::from_bits(0x3ffbf99875f54fab),
        f64::from_bits(0x3bedf7f415771a46),
    ],
    [
        f64::from_bits(0x3fe0bfffff1c6921),
        f64::from_bits(0x3fe277e4598e35d0),
        f64::from_bits(0x3ffbb91f8addd186),
        f64::from_bits(0x3be0f16c5e0e5a73),
    ],
    [
        f64::from_bits(0x3fe0dfffff07154b),
        f64::from_bits(0x3fe2a2a5b6bc3986),
        f64::from_bits(0x3ffb79848301646d),
        f64::from_bits(0xbbf02dd0bbaa5310),
    ],
    [
        f64::from_bits(0x3fe1000002fcdda4),
        f64::from_bits(0x3fe2cd9902a59f1e),
        f64::from_bits(0x3ffb3ac2705219bf),
        f64::from_bits(0xbbe59357112fa616),
    ],
    [
        f64::from_bits(0x3fe1200001ce1140),
        f64::from_bits(0x3fe2f8bedf0a67c2),
        f64::from_bits(0x3ffafcd49ab8ae2a),
        f64::from_bits(0x3be2c5429303f346),
    ],
    [
        f64::from_bits(0x3fe1400004d0f355),
        f64::from_bits(0x3fe3241808fcc7bf),
        f64::from_bits(0x3ffabfb6497b9a36),
        f64::from_bits(0x3bebc044b5a59234),
    ],
    [
        f64::from_bits(0x3fe1600000fb0c8a),
        f64::from_bits(0x3fe34fa52471618b),
        f64::from_bits(0x3ffa83630d26d117),
        f64::from_bits(0xbbdbfbb23f7bb7c9),
    ],
    [
        f64::from_bits(0x3fe18000026f10b3),
        f64::from_bits(0x3fe37b66f7579056),
        f64::from_bits(0x3ffa47d66b4cf4b1),
        f64::from_bits(0x3bf0f6b4af0b5de9),
    ],
    [
        f64::from_bits(0x3fe19ffffd0978f8),
        f64::from_bits(0x3fe3a75e290cc78c),
        f64::from_bits(0x3ffa0d0c36c21315),
        f64::from_bits(0x3beb2129a296b262),
    ],
    [
        f64::from_bits(0x3fe1bffffd94840b),
        f64::from_bits(0x3fe3d38b85b4e4a4),
        f64::from_bits(0x3ff9d30032f2ecef),
        f64::from_bits(0xbbdbab1ab9bb7d74),
    ],
    [
        f64::from_bits(0x3fe1dffffbda1ea1),
        f64::from_bits(0x3fe3ffefbf3cee2f),
        f64::from_bits(0x3ff999ae6770fed8),
        f64::from_bits(0x3bda0bdcb4ace9a4),
    ],
    [
        f64::from_bits(0x3fe1fffffc989533),
        f64::from_bits(0x3fe42c8b9c27900c),
        f64::from_bits(0x3ff96112e0d9f1ac),
        f64::from_bits(0xbbee19eb2fa2d81a),
    ],
    [
        f64::from_bits(0x3fe22000012b8d26),
        f64::from_bits(0x3fe4595fe11975ca),
        f64::from_bits(0x3ff92929cdaa4e80),
        f64::from_bits(0x3bf23382acc82d4b),
    ],
    [
        f64::from_bits(0x3fe2400004f4d6af),
        f64::from_bits(0x3fe4866d4d224131),
        f64::from_bits(0x3ff8f1ef815c34e8),
        f64::from_bits(0xbbd0c6ff3b740a99),
    ],
    [
        f64::from_bits(0x3fe25ffffcc07bda),
        f64::from_bits(0x3fe4b3b498b7d010),
        f64::from_bits(0x3ff8bb6073e7ffa1),
        f64::from_bits(0x3bebc31b1ad7a9c2),
    ],
    [
        f64::from_bits(0x3fe28000042d9639),
        f64::from_bits(0x3fe4e136b64540d1),
        f64::from_bits(0x3ff88578f4374938),
        f64::from_bits(0x3be36de91b85e901),
    ],
    [
        f64::from_bits(0x3fe2a00003be29a0),
        f64::from_bits(0x3fe50ef452bffd96),
        f64::from_bits(0x3ff85035c0042c06),
        f64::from_bits(0x3be15d0176f5efbd),
    ],
    [
        f64::from_bits(0x3fe2bffffaa91f12),
        f64::from_bits(0x3fe53cee3e2f4e0d),
        f64::from_bits(0x3ff81b938542df07),
        f64::from_bits(0x3be555cd17662a2b),
    ],
    [
        f64::from_bits(0x3fe2dffffe884891),
        f64::from_bits(0x3fe56b256c1a2470),
        f64::from_bits(0x3ff7e78ee422ea70),
        f64::from_bits(0x3bf03504bd030c11),
    ],
    [
        f64::from_bits(0x3fe2fffffe87152b),
        f64::from_bits(0x3fe5999a9beaaaa1),
        f64::from_bits(0x3ff7b424d18fe9b3),
        f64::from_bits(0xbb649a5f773e0e64),
    ],
    [
        f64::from_bits(0x3fe31fffffc1a721),
        f64::from_bits(0x3fe5c84eafe0e564),
        f64::from_bits(0x3ff78152338db8d4),
        f64::from_bits(0x3beaf4285da8e935),
    ],
    [
        f64::from_bits(0x3fe33fffff70a372),
        f64::from_bits(0x3fe5f74282191d64),
        f64::from_bits(0x3ff74f141122bcae),
        f64::from_bits(0x3bdb1c4bdee4bfaf),
    ],
    [
        f64::from_bits(0x3fe360000436e836),
        f64::from_bits(0x3fe62676fde6ccff),
        f64::from_bits(0x3ff71d677644252c),
        f64::from_bits(0xbbec3d10e08c3afb),
    ],
    [
        f64::from_bits(0x3fe37ffffcbe9641),
        f64::from_bits(0x3fe655ecee9ffdaf),
        f64::from_bits(0x3ff6ec49a6fc0515),
        f64::from_bits(0x3bdda4532ed29567),
    ],
    [
        f64::from_bits(0x3fe39fffffb6d6ca),
        f64::from_bits(0x3fe685a55e67a1e1),
        f64::from_bits(0x3ff6bbb7bc2ae969),
        f64::from_bits(0x3becbf7b2ef43882),
    ],
    [
        f64::from_bits(0x3fe3c00004934fec),
        f64::from_bits(0x3fe6b5a12cc07d75),
        f64::from_bits(0x3ff68baf10b02ef8),
        f64::from_bits(0xbbe7c8fbfeb7cabd),
    ],
    [
        f64::from_bits(0x3fe3e00003f5cf7f),
        f64::from_bits(0x3fe6e5e13e59def6),
        f64::from_bits(0x3ff65c2d0e61500f),
        f64::from_bits(0xbbe30ba4035f7845),
    ],
    [
        f64::from_bits(0x3fe4000005280ad9),
        f64::from_bits(0x3fe7166691ab4c3e),
        f64::from_bits(0x3ff62d2f19f01c90),
        f64::from_bits(0xbbf1e9f5ffe95f6a),
    ],
    [
        f64::from_bits(0x3fe42000049efb65),
        f64::from_bits(0x3fe7473218af3b9d),
        f64::from_bits(0x3ff5feb2b86465e4),
        f64::from_bits(0x3bc4cad7280d591e),
    ],
    [
        f64::from_bits(0x3fe440000035ccb6),
        f64::from_bits(0x3fe77844cb4ff1e5),
        f64::from_bits(0x3ff5d0b57c455428),
        f64::from_bits(0x3bed8c187ba5617c),
    ],
    [
        f64::from_bits(0x3fe4600003346717),
        f64::from_bits(0x3fe7a99fba258778),
        f64::from_bits(0x3ff5a334f4392254),
        f64::from_bits(0xbbefd14afc84a570),
    ],
    [
        f64::from_bits(0x3fe4800003002575),
        f64::from_bits(0x3fe7db43d836768f),
        f64::from_bits(0x3ff5762edcf97e0c),
        f64::from_bits(0xbbdd7eba5f5df49e),
    ],
    [
        f64::from_bits(0x3fe4a000055bf381),
        f64::from_bits(0x3fe80d3235edeefa),
        f64::from_bits(0x3ff549a0ea46e31f),
        f64::from_bits(0xbbdba52276823eac),
    ],
    [
        f64::from_bits(0x3fe4c00004ce10e3),
        f64::from_bits(0x3fe83f6bd67dc1a8),
        f64::from_bits(0x3ff51d88ed82bcc4),
        f64::from_bits(0xbbeae92d077d29ea),
    ],
    [
        f64::from_bits(0x3fe4e000016c60e1),
        f64::from_bits(0x3fe871f1ca0aaf31),
        f64::from_bits(0x3ff4f1e4bdacbf16),
        f64::from_bits(0x3be8295846ee425e),
    ],
    [
        f64::from_bits(0x3fe4ffffff966f0a),
        f64::from_bits(0x3fe8a4c52bff2dae),
        f64::from_bits(0x3ff4c6b23917657e),
        f64::from_bits(0xbbf127c25c86c705),
    ],
    [
        f64::from_bits(0x3fe520000076e6eb),
        f64::from_bits(0x3fe8d7e7175651e8),
        f64::from_bits(0x3ff49bef4f459b05),
        f64::from_bits(0xbbb1e9d14181bbfc),
    ],
    [
        f64::from_bits(0x3fe5400003d12d3b),
        f64::from_bits(0x3fe90b58a976ed56),
        f64::from_bits(0x3ff47199fdf24af4),
        f64::from_bits(0x3be38c17c30decaf),
    ],
    [
        f64::from_bits(0x3fe55ffffce7fa8d),
        f64::from_bits(0x3fe93f1af03a3a09),
        f64::from_bits(0x3ff447b05f13234b),
        f64::from_bits(0x3bf1b8b270df7e20),
    ],
    [
        f64::from_bits(0x3fe580000331b46a),
        f64::from_bits(0x3fe9732f38e83134),
        f64::from_bits(0x3ff41e3068d8b41b),
        f64::from_bits(0xbbee24d8b90bc28b),
    ],
    [
        f64::from_bits(0x3fe59ffffc14848e),
        f64::from_bits(0x3fe9a7968471b489),
        f64::from_bits(0x3ff3f5185de3aa73),
        f64::from_bits(0xbbecacd9e0761536),
    ],
    [
        f64::from_bits(0x3fe5bffffb7cd395),
        f64::from_bits(0x3fe9dc5224a8b955),
        f64::from_bits(0x3ff3cc664f8fff15),
        f64::from_bits(0xbbf67c9782045611),
    ],
    [
        f64::from_bits(0x3fe5e000000dcc40),
        f64::from_bits(0x3fea11634df5b93e),
        f64::from_bits(0x3ff3a41875853228),
        f64::from_bits(0xbbf585dad481f350),
    ],
    [
        f64::from_bits(0x3fe6000002efd2fc),
        f64::from_bits(0x3fea46cb30d16323),
        f64::from_bits(0x3ff37c2d187962ae),
        f64::from_bits(0x3bf004c3a5f77bb0),
    ],
    [
        f64::from_bits(0x3fe61ffffeb8088a),
        f64::from_bits(0x3fea7c8b053920c0),
        f64::from_bits(0x3ff354a2891769a9),
        f64::from_bits(0x3bbc6b303fee3029),
    ],
    [
        f64::from_bits(0x3fe6400000f3ca06),
        f64::from_bits(0x3feab2a428a1911a),
        f64::from_bits(0x3ff32d770a6f0a4a),
        f64::from_bits(0x3bf2a6f8fac5081a),
    ],
    [
        f64::from_bits(0x3fe65ffffe9ec2f4),
        f64::from_bits(0x3feae917d4ce7239),
        f64::from_bits(0x3ff306a90751a948),
        f64::from_bits(0xbbe950b551ab9dbd),
    ],
    [
        f64::from_bits(0x3fe6800003d43966),
        f64::from_bits(0x3feb1fe7708b998a),
        f64::from_bits(0x3ff2e036d7a153c7),
        f64::from_bits(0x3bdd36e2a1e4a14e),
    ],
    [
        f64::from_bits(0x3fe69ffffab67783),
        f64::from_bits(0x3feb57142e575464),
        f64::from_bits(0x3ff2ba1f05006cb6),
        f64::from_bits(0x3bea9a4a473c2e31),
    ],
    [
        f64::from_bits(0x3fe6bffffcb65f89),
        f64::from_bits(0x3feb8e9f981efd2f),
        f64::from_bits(0x3ff2945fe948d9f7),
        f64::from_bits(0xbbca5294e802df72),
    ],
    [
        f64::from_bits(0x3fe6dffffc5609a9),
        f64::from_bits(0x3febc68afaed6ff1),
        f64::from_bits(0x3ff26ef81533411e),
        f64::from_bits(0xbbf89153f51bc566),
    ],
    [
        f64::from_bits(0x3fe6fffffc4eef86),
        f64::from_bits(0x3febfed7c62205fe),
        f64::from_bits(0x3ff249e60e70978c),
        f64::from_bits(0x3bc39021a2b9ff56),
    ],
    [
        f64::from_bits(0x3fe72000004d98b3),
        f64::from_bits(0x3fec3787716968ad),
        f64::from_bits(0x3ff2252861be7751),
        f64::from_bits(0x3befc9c574ee2211),
    ],
    [
        f64::from_bits(0x3fe73ffffc155075),
        f64::from_bits(0x3fec709b5ec6fd4e),
        f64::from_bits(0x3ff200bdb5d53311),
        f64::from_bits(0x3be28a4da269ae63),
    ],
    [
        f64::from_bits(0x3fe760000498c203),
        f64::from_bits(0x3fecaa15323d08c1),
        f64::from_bits(0x3ff1dca493433f65),
        f64::from_bits(0x3bf8cae414a28fb7),
    ],
    [
        f64::from_bits(0x3fe77fffff1e5636),
        f64::from_bits(0x3fece3f64147c12c),
        f64::from_bits(0x3ff1b8dbbfe294a8),
        f64::from_bits(0xbbe7e19c4b56a744),
    ],
    [
        f64::from_bits(0x3fe7a0000226d45a),
        f64::from_bits(0x3fed1e404120eb7f),
        f64::from_bits(0x3ff19561d15f8278),
        f64::from_bits(0x3be64b28032c5d4c),
    ],
    [
        f64::from_bits(0x3fe7c0000250a5aa),
        f64::from_bits(0x3fed58f4b112a1e1),
        f64::from_bits(0x3ff172358a59d565),
        f64::from_bits(0xbbe716deb8dc7867),
    ],
    [
        f64::from_bits(0x3fe7e0000482f82e),
        f64::from_bits(0x3fed94153576bdf0),
        f64::from_bits(0x3ff14f55a22a1c5b),
        f64::from_bits(0x3bf207e1e1305604),
    ],
    [
        f64::from_bits(0x3fe800000205003e),
        f64::from_bits(0x3fedcfa364d69ff7),
        f64::from_bits(0x3ff12cc0e37eb26f),
        f64::from_bits(0xbbd52ec6e32395f8),
    ],
    [
        f64::from_bits(0x3fe81ffffbf99411),
        f64::from_bits(0x3fee0ba0ebf98f51),
        f64::from_bits(0x3ff10a7616ddd5d6),
        f64::from_bits(0xbbece0d659866045),
    ],
    [
        f64::from_bits(0x3fe840000248e3a3),
        f64::from_bits(0x3fee480f9bb7f565),
        f64::from_bits(0x3ff0e873fb84e05c),
        f64::from_bits(0x3bf4e5e81595df92),
    ],
    [
        f64::from_bits(0x3fe860000145c157),
        f64::from_bits(0x3fee84f10a10b3ab),
        f64::from_bits(0x3ff0c6b97cbd7b1e),
        f64::from_bits(0xbbe19de6d5f121d0),
    ],
    [
        f64::from_bits(0x3fe88000022631b9),
        f64::from_bits(0x3feec2470be1f047),
        f64::from_bits(0x3ff0a5456d0b3ee6),
        f64::from_bits(0xbbc272b1a3ba2c6f),
    ],
    [
        f64::from_bits(0x3fe8a000045f7828),
        f64::from_bits(0x3fef00136c45ba1c),
        f64::from_bits(0x3ff08416af2a0f09),
        f64::from_bits(0x3be82b565b63c799),
    ],
    [
        f64::from_bits(0x3fe8bfffffc686cf),
        f64::from_bits(0x3fef3e57f03c824b),
        f64::from_bits(0x3ff0632c33502220),
        f64::from_bits(0xbbd039ad2dbeeb25),
    ],
    [
        f64::from_bits(0x3fe8dffffd8644c6),
        f64::from_bits(0x3fef7d168774261d),
        f64::from_bits(0x3ff04284dd5b3019),
        f64::from_bits(0x3bd79f33e1eba933),
    ],
    [
        f64::from_bits(0x3fe8fffffe4e7937),
        f64::from_bits(0x3fefbc511a99a641),
        f64::from_bits(0x3ff0221f9f69840b),
        f64::from_bits(0xbbea9e847beee018),
    ],
    [
        f64::from_bits(0x3fe920000435251f),
        f64::from_bits(0x3feffc099eb22390),
        f64::from_bits(0x3ff001fb6f7c51e8),
        f64::from_bits(0xbb5a12e731032e0a),
    ],
];
/// `EADD(x, y)` → `(z, zz)`.
#[inline]
fn eadd(x: f64, y: f64) -> (f64, f64) {
    let z = x + y;
    let zz = if x.abs() > y.abs() {
        (x - z) + y
    } else {
        (y - z) + x
    };
    (z, zz)
}

/// `1 / (b + db)` as `DIV2(1.0, 0.0, b, db, …)` with the FMA `MUL12`, → `c + dc`.
/// (Whether `c·db` is fused is inert here — measured on 25M inputs; the rest of
/// `DIV2` is not.)
#[inline]
fn div2_one(b: f64, db: f64) -> f64 {
    let c = 1.0 / b;
    let u = c * b;
    let uu = fma(c, b, -u);
    let cc = fma(-c, db, ((1.0 - u) - uu) + 0.0) / b;
    let z = c + cc;
    let zz = (c - z) + cc;
    z + zz
}

/// `tan(a + da)` for `|a| ≤ 0.0608` (cases VI, VIII, X); `n` odd → `−cot`.
fn tan_small(a: f64, da: f64, n: i32) -> f64 {
    let a2 = a * a;
    let mut t2 = fma(a2, T_D11, T_D9);
    t2 = fma(a2, t2, T_D7);
    t2 = fma(a2, t2, T_D5);
    t2 = fma(a2, t2, T_D3);
    let t2 = fma(a * a2, t2, da);
    if n != 0 {
        let (b, db) = eadd(a, t2);
        -div2_one(b, db)
    } else {
        a + t2
    }
}

/// `tan` / `−cot` of `±(ya + yya)` from the `xfg` table (cases VII, IX, XI).
fn tan_table(ya: f64, yya: f64, sy: f64, n: i32) -> f64 {
    let i = fma(256.0, ya, T_MFFTNHF) as i32 as usize;
    let z = (ya - XFG[i][0]) + yya;
    let z2 = z * z;
    let pz = fma(z * z2, fma(z2, T_E1, T_E0), z);
    let (fi, gi) = (XFG[i][1], XFG[i][2]);
    if n != 0 {
        let t2 = pz * (fi + gi) / (fi + pz);
        let y = gi - t2;
        -sy * y
    } else {
        let t2 = pz * (gi + fi) / (gi - pz);
        let y = fi + t2;
        sy * y
    }
}

/// The reduced `a + da` split by sign → `tan` of it (or `−cot` for odd `n`).
fn tan_reduced(a: f64, da: f64, n: i32) -> f64 {
    let (ya, yya, sy) = if a < 0.0 {
        (-a, -da, -1.0)
    } else {
        (a, da, 1.0)
    };
    if ya <= T_GY2 {
        return tan_small(a, da, n);
    }
    tan_table(ya, yya, sy, n)
}

/// glibc `tan(x)` (`s_tan.c` as `__tan_fma`).
pub fn tan(x: f64) -> f64 {
    let ux = (x.to_bits() >> 32) as u32;
    if (ux & 0x7ff0_0000) == 0x7ff0_0000 {
        return x - x;
    }
    let w = x.abs();
    if w <= T_G1 {
        return x;
    }
    if w <= T_G2 {
        let x2 = x * x;
        let mut t2 = fma(x2, T_D11, T_D9);
        t2 = fma(x2, t2, T_D7);
        t2 = fma(x2, t2, T_D5);
        t2 = fma(x2, t2, T_D3);
        return fma(t2, x * x2, x);
    }
    if w <= T_G3 {
        let i = fma(256.0, w, T_MFFTNHF) as i32 as usize;
        let z = w - XFG[i][0];
        let z2 = z * z;
        let s = if x < 0.0 { -1.0 } else { 1.0 };
        let pz = fma(z * z2, fma(z2, T_E1, T_E0), z);
        let (fi, gi) = (XFG[i][1], XFG[i][2]);
        let t2 = pz * (gi + fi) / (gi - pz);
        let y = fi + t2;
        return s * y;
    }
    if w <= T_G4 {
        let t = fma(x, T_HPINV, T_TOINT);
        let xn = t - T_TOINT;
        let t1 = fma(-xn, T_MP2, fma(-xn, T_MP1, x));
        let n = (t.to_bits() as u32 & 1) as i32;
        // `da = xn·mp3` has two subtract uses: fused into each
        let a = fma(-xn, T_MP3, t1);
        let da = fma(-xn, T_MP3, t1 - a);
        return tan_reduced(a, da, n);
    }
    if w <= T_G5 {
        let t = fma(x, T_HPINV, T_TOINT);
        let xn = t - T_TOINT;
        let t1 = fma(-xn, T_MP2, fma(-xn, T_MP1, x));
        let n = (t.to_bits() as u32 & 1) as i32;
        let tt = fma(-xn, T_PP3, t1);
        let da = fma(-xn, T_PP3, t1 - tt);
        let a = fma(-xn, T_PP4, tt);
        let da = fma(-xn, T_PP4, tt - a) + da;
        let (a, da) = eadd(a, da);
        return tan_reduced(a, da, n);
    }
    let (n, a, da) = branred(x);
    let (a, da) = eadd(a, da);
    tan_reduced(a, da, n & 1)
}
