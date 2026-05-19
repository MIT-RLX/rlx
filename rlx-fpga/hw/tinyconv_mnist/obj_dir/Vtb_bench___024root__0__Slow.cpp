// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_bench.h for the primary calling header

#include "Vtb_bench__pch.h"

void Vtb_bench___024root___timing_ready(Vtb_bench___024root* vlSelf);

VL_ATTR_COLD void Vtb_bench___024root___eval_static(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_static\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.tb_bench__DOT__clk = 0U;
    vlSelfRef.tb_bench__DOT__rst = 1U;
    vlSelfRef.tb_bench__DOT__start = 0U;
    vlSelfRef.tb_bench__DOT__in_addr = 0U;
    vlSelfRef.tb_bench__DOT__in_we = 0U;
    vlSelfRef.tb_bench__DOT__in_din = 0U;
    vlSelfRef.tb_bench__DOT__cycles_counter = 0ULL;
    vlSelfRef.tb_bench__DOT__counting = 0U;
    vlSelfRef.__VactTriggered[0U] = (4ULL | vlSelfRef.__VactTriggered[0U]);
    vlSelfRef.__Vtrigprevexpr___TOP__tb_bench__DOT__clk__0 = 0U;
    vlSelfRef.__Vtrigprevexpr___TOP__tb_bench__DOT__done__0 
        = vlSelfRef.tb_bench__DOT__done;
    Vtb_bench___024root___timing_ready(vlSelf);
    do {
        vlSelfRef.__VactTriggeredAcc[vlSelfRef.__Vi] 
            = vlSelfRef.__VactTriggered[vlSelfRef.__Vi];
        vlSelfRef.__Vi = ((IData)(1U) + vlSelfRef.__Vi);
    } while ((0U >= vlSelfRef.__Vi));
}

VL_ATTR_COLD void Vtb_bench___024root___eval_static__TOP(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_static__TOP\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.tb_bench__DOT__clk = 0U;
    vlSelfRef.tb_bench__DOT__rst = 1U;
    vlSelfRef.tb_bench__DOT__start = 0U;
    vlSelfRef.tb_bench__DOT__in_addr = 0U;
    vlSelfRef.tb_bench__DOT__in_we = 0U;
    vlSelfRef.tb_bench__DOT__in_din = 0U;
    vlSelfRef.tb_bench__DOT__cycles_counter = 0ULL;
    vlSelfRef.tb_bench__DOT__counting = 0U;
}

VL_ATTR_COLD void Vtb_bench___024root___eval_initial__TOP(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_initial__TOP\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    VL_READMEM_N(true, 8, 72, 0, "weights/conv1_w.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_w_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 8, 0, "weights/conv1_b.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_b_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 8, 0, "weights/conv1_m0.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_m0_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 8, 0, "weights/conv1_sh.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_sh_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 1152, 0, "weights/conv2_w.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_w_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 16, 0, "weights/conv2_b.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_b_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 16, 0, "weights/conv2_m0.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_m0_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 16, 0, "weights/conv2_sh.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_sh_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 4000, 0, "weights/fc_w.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_w_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 10, 0, "weights/fc_b.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_b_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 32, 10, 0, "weights/fc_m0.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_m0_rom__DOT__mem)
                 , 0, ~0ULL);
    VL_READMEM_N(true, 8, 10, 0, "weights/fc_sh.mem"s
                 ,  &(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_sh_rom__DOT__mem)
                 , 0, ~0ULL);
}

VL_ATTR_COLD void Vtb_bench___024root___eval_final(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_final\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_bench___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vtb_bench___024root___eval_phase__stl(Vtb_bench___024root* vlSelf);

VL_ATTR_COLD void Vtb_bench___024root___eval_settle(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_settle\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VstlIterCount;
    // Body
    __VstlIterCount = 0U;
    vlSelfRef.__VstlFirstIteration = 1U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VstlIterCount)))) {
#ifdef VL_DEBUG
            Vtb_bench___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
#endif
            VL_FATAL_MT("tb_bench.sv", 4, "", "DIDNOTCONVERGE: Settle region did not converge after '--converge-limit' of 10000 tries");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        vlSelfRef.__VstlPhaseResult = Vtb_bench___024root___eval_phase__stl(vlSelf);
        vlSelfRef.__VstlFirstIteration = 0U;
    } while (vlSelfRef.__VstlPhaseResult);
}

VL_ATTR_COLD void Vtb_bench___024root___eval_triggers_vec__stl(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_triggers_vec__stl\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VstlTriggered[0U] = ((0xfffffffffffffffeULL 
                                      & vlSelfRef.__VstlTriggered[0U]) 
                                     | (IData)((IData)(vlSelfRef.__VstlFirstIteration)));
}

VL_ATTR_COLD bool Vtb_bench___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_bench___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(Vtb_bench___024root___trigger_anySet__stl(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD bool Vtb_bench___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___trigger_anySet__stl\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        if (in[n]) {
            return (1U);
        }
        n = ((IData)(1U) + n);
    } while ((1U > n));
    return (0U);
}

extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_had1126da_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_h21db8459_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_h0e45ff86_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_h392ef149_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_h8ddecfd7_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_hce4ff95b_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_hd55a8f63_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_hd25ea8a5_0;
extern const VlUnpacked<CData/*0:0*/, 64> Vtb_bench__ConstPool__TABLE_hf85a0a8a_0;

VL_ATTR_COLD void Vtb_bench___024root___stl_sequent__TOP__0(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___stl_sequent__TOP__0\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out;
    tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out = 0;
    IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out;
    tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out = 0;
    IData/*31:0*/ tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out;
    tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out = 0;
    CData/*5:0*/ __Vtableidx1;
    __Vtableidx1 = 0;
    QData/*63:0*/ __VdfgRegularize_h6e95ff9d_0_0;
    __VdfgRegularize_h6e95ff9d_0_0 = 0;
    QData/*63:0*/ __VdfgRegularize_h6e95ff9d_0_1;
    __VdfgRegularize_h6e95ff9d_0_1 = 0;
    QData/*63:0*/ __VdfgRegularize_h6e95ff9d_0_2;
    __VdfgRegularize_h6e95ff9d_0_2 = 0;
    QData/*63:0*/ __VdfgRegularize_h6e95ff9d_0_4;
    __VdfgRegularize_h6e95ff9d_0_4 = 0;
    QData/*63:0*/ __VdfgRegularize_h6e95ff9d_0_6;
    __VdfgRegularize_h6e95ff9d_0_6 = 0;
    QData/*63:0*/ __VdfgRegularize_h6e95ff9d_0_8;
    __VdfgRegularize_h6e95ff9d_0_8 = 0;
    // Body
    vlSelfRef.tb_bench__DOT__u_top__DOT__l7_y_we = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__w_addr 
        = (0x00000fffU & (((IData)(0x00000190U) * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i)) 
                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i)));
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__w_addr 
        = (0x0000007fU & (((IData)(3U) * (((IData)(3U) 
                                           * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc)) 
                                          + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i))) 
                          + ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic) 
                             + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i))));
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__w_addr 
        = (0x000007ffU & ((0x000007f8U & ((((IData)(3U) 
                                            * (((IData)(3U) 
                                                * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc)) 
                                               + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i))) 
                                           + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i)) 
                                          << 3U)) + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic)));
    vlSelfRef.tb_bench__DOT__u_top__DOT__l6_y_we = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__a0_addr = 
        (0x000003ffU & ((IData)(vlSelfRef.tb_bench__DOT__start)
                         ? ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic) 
                            + (((IData)(0x0000001cU) 
                                * ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i) 
                                   + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh))) 
                               + ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow) 
                                  + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i))))
                         : (IData)(vlSelfRef.tb_bench__DOT__in_addr)));
    vlSelfRef.tb_bench__DOT__u_top__DOT__l0_y_we = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l3_y_we = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l5_y_we = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l1_y_we = 0U;
    if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state) 
                  >> 2U)))) {
        if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l1_y_we = 1U;
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__l4_y_we = 0U;
    if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state) 
                  >> 2U)))) {
        if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))) {
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l4_y_we = 1U;
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__l2_y_we = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l0_done = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l1_done = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l2_done = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l3_done = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l4_done = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l5_done = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l6_done = 0U;
    vlSelfRef.tb_bench__DOT__u_top__DOT__l7_done = 0U;
    __VdfgRegularize_h6e95ff9d_0_4 = VL_MULS_QQQ(64, 
                                                 VL_EXTENDS_QI(64,32, vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc), 
                                                 VL_EXTENDS_QI(64,32, vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__m0_dout));
    __VdfgRegularize_h6e95ff9d_0_6 = VL_MULS_QQQ(64, 
                                                 VL_EXTENDS_QI(64,32, vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc), 
                                                 VL_EXTENDS_QI(64,32, vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__m0_dout));
    __VdfgRegularize_h6e95ff9d_0_8 = VL_MULS_QQQ(64, 
                                                 VL_EXTENDS_QI(64,32, vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__acc), 
                                                 VL_EXTENDS_QI(64,32, vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m0_dout));
    __Vtableidx1 = (((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage) 
                     << 2U) | (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__cstate));
    vlSelfRef.tb_bench__DOT__u_top__DOT__l0_start = Vtb_bench__ConstPool__TABLE_had1126da_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l1_start = Vtb_bench__ConstPool__TABLE_h21db8459_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l2_start = Vtb_bench__ConstPool__TABLE_h0e45ff86_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l3_start = Vtb_bench__ConstPool__TABLE_h392ef149_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l4_start = Vtb_bench__ConstPool__TABLE_h8ddecfd7_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l5_start = Vtb_bench__ConstPool__TABLE_hce4ff95b_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l6_start = Vtb_bench__ConstPool__TABLE_hd55a8f63_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__u_top__DOT__l7_start = Vtb_bench__ConstPool__TABLE_hd25ea8a5_0
        [__Vtableidx1];
    vlSelfRef.tb_bench__DOT__done = Vtb_bench__ConstPool__TABLE_hf85a0a8a_0
        [__Vtableidx1];
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__l6_y_we = 1U;
                }
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a7_addr = 
        ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l6_y_we)
          ? (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i)
          : (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__i));
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__l0_y_we = 1U;
                }
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a1_addr = 
        (0x00001fffU & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l0_y_we)
                         ? ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc) 
                            + (0x00001ff8U & ((((IData)(0x0000001aU) 
                                                * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh)) 
                                               + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow)) 
                                              << 3U)))
                         : (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__i)));
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__l3_y_we = 1U;
                }
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a4_addr = 
        (0x000007ffU & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l3_y_we)
                         ? ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc) 
                            + (0x000007f0U & ((((IData)(0x0000000bU) 
                                                * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh)) 
                                               + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow)) 
                                              << 4U)))
                         : (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__i)));
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state) 
                      >> 1U)))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state)))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l5_y_we = 1U;
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a6_addr = 
        (0x000001ffU & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l5_y_we)
                         ? ((0x000001f0U & ((((IData)(5U) 
                                              * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh)) 
                                             + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow)) 
                                            << 4U)) 
                            + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc))
                         : (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i)));
    vlSelfRef.tb_bench__DOT__u_top__DOT__a2_addr = 
        (0x00001fffU & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l1_y_we)
                         ? (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__i)
                         : ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc) 
                            + (0x00001ff8U & ((((IData)(0x0000001aU) 
                                                * (
                                                   VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh), 1U) 
                                                   + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i))) 
                                               + (VL_SHIFTL_III(10,32,32, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow), 1U) 
                                                  + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i))) 
                                              << 3U)))));
    vlSelfRef.tb_bench__DOT__u_top__DOT__a5_addr = 
        (0x000007ffU & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l4_y_we)
                         ? (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__i)
                         : ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc) 
                            + (0x000007f0U & ((((IData)(0x0000000bU) 
                                                * (
                                                   VL_SHIFTL_III(32,32,32, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh), 1U) 
                                                   + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i))) 
                                               + (VL_SHIFTL_III(7,32,32, (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow), 1U) 
                                                  + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i))) 
                                              << 4U)))));
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state) 
                      >> 1U)))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state)))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l2_y_we = 1U;
            }
        }
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__a3_addr = 
        (0x000007ffU & ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l2_y_we)
                         ? ((0x000007f8U & ((((IData)(0x0000000dU) 
                                              * (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh)) 
                                             + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow)) 
                                            << 3U)) 
                            + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc))
                         : ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic) 
                            + (0x000007f8U & (((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow) 
                                               + (((IData)(0x0000000dU) 
                                                   * 
                                                   ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i) 
                                                    + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh))) 
                                                  + (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i))) 
                                              << 3U)))));
    vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = vlSelfRef.tb_bench__DOT__u_top__DOT__cstate;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state) 
                      >> 1U)))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state)))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l1_done = 1U;
            }
        }
    }
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state) 
                      >> 1U)))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state)))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l4_done = 1U;
            }
        }
    }
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state) 
                      >> 1U)))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state)))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l7_y_we = 1U;
            }
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l7_done = 1U;
            }
        }
    }
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__l6_done = 1U;
                }
            }
        }
    }
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state) 
                      >> 1U)))) {
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l2_done = 1U;
            }
        }
    }
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state) 
                      >> 1U)))) {
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__l5_done = 1U;
            }
        }
    }
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__l0_done = 1U;
                }
            }
        }
    }
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__l3_done = 1U;
                }
            }
        }
    }
    if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__cstate))) {
        if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__cstate))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__start)))) {
                vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 0U;
            }
        } else {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext 
                = ((7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage))
                    ? 3U : 1U);
        }
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__cstate))) {
        if (((0U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
             & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l0_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((1U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l1_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l2_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((3U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l3_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((4U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l4_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((5U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l5_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((6U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l6_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        } else if (((7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__stage)) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l7_done))) {
            vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 2U;
        }
    } else if (vlSelfRef.tb_bench__DOT__start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__cnext = 1U;
    }
    __VdfgRegularize_h6e95ff9d_0_0 = (__VdfgRegularize_h6e95ff9d_0_4 
                                      + (VL_LTES_IQQ(64, 0ULL, __VdfgRegularize_h6e95ff9d_0_4)
                                          ? 0x0000000040000000ULL
                                          : 0xffffffffc0000001ULL));
    __VdfgRegularize_h6e95ff9d_0_1 = (__VdfgRegularize_h6e95ff9d_0_6 
                                      + (VL_LTES_IQQ(64, 0ULL, __VdfgRegularize_h6e95ff9d_0_6)
                                          ? 0x0000000040000000ULL
                                          : 0xffffffffc0000001ULL));
    __VdfgRegularize_h6e95ff9d_0_2 = (__VdfgRegularize_h6e95ff9d_0_8 
                                      + (VL_LTES_IQQ(64, 0ULL, __VdfgRegularize_h6e95ff9d_0_8)
                                          ? 0x0000000040000000ULL
                                          : 0xffffffffc0000001ULL));
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state) 
                      >> 1U)))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state)))) {
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l1_start)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__next = 0U;
                }
            }
        }
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))
                ? ((0x151fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__i))
                    ? 4U : 1U) : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l1_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu1__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state) 
                      >> 1U)))) {
            if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state)))) {
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l4_start)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__next = 0U;
                }
            }
        }
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))
                ? ((0x078fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__i))
                    ? 4U : 1U) : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l4_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_relu2__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state) 
                      >> 1U)))) {
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))) {
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l7_start)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next = 0U;
                }
            } else {
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next = 5U;
            }
        }
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))
                ? ((9U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__i))
                    ? 4U : 1U) : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l7_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_argmax__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state;
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
                    if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l6_start)))) {
                        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next = 0U;
                    }
                } else {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next 
                        = ((9U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i))
                            ? 0x0bU : 1U);
                }
            } else {
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next 
                    = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))
                        ? 0x0aU : 9U);
            }
        }
    } else if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next 
            = ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))
                ? ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))
                    ? 8U : ((0x018fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i))
                             ? 7U : 4U)) : ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))
                                             ? 6U : 5U));
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))
                ? 4U : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l6_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state) 
                      >> 1U)))) {
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l2_start)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next = 0U;
                }
            } else {
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next 
                    = ((((0x0cU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh)) 
                         & (0x0cU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow))) 
                        & (7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc)))
                        ? 5U : 1U);
            }
        }
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))
                ? (((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i))
                    ? 4U : 1U) : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l2_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool1__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state;
    if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state) 
                      >> 1U)))) {
            if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
                if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l5_start)))) {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next = 0U;
                }
            } else {
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next 
                    = ((((4U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh)) 
                         & (4U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow))) 
                        & (0x0fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc)))
                        ? 5U : 1U);
            }
        }
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))
                ? (((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i) 
                    & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i))
                    ? 4U : 1U) : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l5_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_pool2__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state;
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
                    if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l0_start)))) {
                        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next = 0U;
                    }
                } else {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next 
                        = ((((0x19U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh)) 
                             & (0x19U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow))) 
                            & (7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc)))
                            ? 0x0bU : 1U);
                }
            } else {
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next 
                    = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))
                        ? 0x0aU : 9U);
            }
        }
    } else if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next 
            = ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))
                ? ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))
                    ? 8U : ((((2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i)) 
                              & (2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i))) 
                             & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic)))
                             ? 7U : 4U)) : ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))
                                             ? 6U : 5U));
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))
                ? 4U : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l0_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__next = 1U;
    }
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next 
        = vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state;
    if ((8U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
        if ((1U & (~ ((IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state) 
                      >> 2U)))) {
            if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
                if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
                    if ((1U & (~ (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__l3_start)))) {
                        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next = 0U;
                    }
                } else {
                    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next 
                        = ((((0x0aU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh)) 
                             & (0x0aU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow))) 
                            & (0x0fU == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc)))
                            ? 0x0bU : 1U);
                }
            } else {
                vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next 
                    = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))
                        ? 0x0aU : 9U);
            }
        }
    } else if ((4U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next 
            = ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))
                ? ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))
                    ? 8U : ((((2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i)) 
                              & (2U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i))) 
                             & (7U == (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic)))
                             ? 7U : 4U)) : ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))
                                             ? 6U : 5U));
    } else if ((2U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next 
            = ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))
                ? 4U : 3U);
    } else if ((1U & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__state))) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next = 2U;
    } else if (vlSelfRef.tb_bench__DOT__u_top__DOT__l3_start) {
        vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__next = 1U;
    }
    tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out 
        = (((0x80000000U == vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc) 
            & (0x80000000U == vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__m0_dout))
            ? 0x7fffffffU : (VL_LTES_IQQ(64, 0ULL, __VdfgRegularize_h6e95ff9d_0_0)
                              ? (IData)((__VdfgRegularize_h6e95ff9d_0_0 
                                         >> 0x0000001fU))
                              : (IData)((- VL_SHIFTRS_QQI(64,64,32, 
                                                          (- __VdfgRegularize_h6e95ff9d_0_0), 0x0000001fU)))));
    tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out 
        = (((0x80000000U == vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc) 
            & (0x80000000U == vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__m0_dout))
            ? 0x7fffffffU : (VL_LTES_IQQ(64, 0ULL, __VdfgRegularize_h6e95ff9d_0_1)
                              ? (IData)((__VdfgRegularize_h6e95ff9d_0_1 
                                         >> 0x0000001fU))
                              : (IData)((- VL_SHIFTRS_QQI(64,64,32, 
                                                          (- __VdfgRegularize_h6e95ff9d_0_1), 0x0000001fU)))));
    tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out 
        = (((0x80000000U == vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__acc) 
            & (0x80000000U == vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__m0_dout))
            ? 0x7fffffffU : (VL_LTES_IQQ(64, 0ULL, __VdfgRegularize_h6e95ff9d_0_2)
                              ? (IData)((__VdfgRegularize_h6e95ff9d_0_2 
                                         >> 0x0000001fU))
                              : (IData)((- VL_SHIFTRS_QQI(64,64,32, 
                                                          (- __VdfgRegularize_h6e95ff9d_0_2), 0x0000001fU)))));
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__rdpot_out 
        = ((0U == (0x0000001fU & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__sh_dout)))
            ? tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out
            : (VL_SHIFTRS_III(32,32,5, tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out, 
                              (0x0000001fU & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__sh_dout))) 
               + (1U & (- (IData)(VL_GTS_III(32, (tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out 
                                                  & (((IData)(1U) 
                                                      << 
                                                      (0x0000001fU 
                                                       & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__sh_dout))) 
                                                     - (IData)(1U))), 
                                             (VL_SHIFTRS_III(32,32,32, 
                                                             (((IData)(1U) 
                                                               << 
                                                               (0x0000001fU 
                                                                & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv1__DOT__sh_dout))) 
                                                              - (IData)(1U)), 1U) 
                                              + (1U 
                                                 & (- (IData)(
                                                              VL_GTS_III(32, 0U, tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__srdhm_out)))))))))));
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__rdpot_out 
        = ((0U == (0x0000001fU & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__sh_dout)))
            ? tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out
            : (VL_SHIFTRS_III(32,32,5, tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out, 
                              (0x0000001fU & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__sh_dout))) 
               + (1U & (- (IData)(VL_GTS_III(32, (tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out 
                                                  & (((IData)(1U) 
                                                      << 
                                                      (0x0000001fU 
                                                       & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__sh_dout))) 
                                                     - (IData)(1U))), 
                                             (VL_SHIFTRS_III(32,32,32, 
                                                             (((IData)(1U) 
                                                               << 
                                                               (0x0000001fU 
                                                                & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_conv2__DOT__sh_dout))) 
                                                              - (IData)(1U)), 1U) 
                                              + (1U 
                                                 & (- (IData)(
                                                              VL_GTS_III(32, 0U, tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__srdhm_out)))))))))));
    vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__rdpot_out 
        = ((0U == (0x0000001fU & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__sh_dout)))
            ? tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out
            : (VL_SHIFTRS_III(32,32,5, tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out, 
                              (0x0000001fU & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__sh_dout))) 
               + (1U & (- (IData)(VL_GTS_III(32, (tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out 
                                                  & (((IData)(1U) 
                                                      << 
                                                      (0x0000001fU 
                                                       & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__sh_dout))) 
                                                     - (IData)(1U))), 
                                             (VL_SHIFTRS_III(32,32,32, 
                                                             (((IData)(1U) 
                                                               << 
                                                               (0x0000001fU 
                                                                & (IData)(vlSelfRef.tb_bench__DOT__u_top__DOT__u_fc__DOT__sh_dout))) 
                                                              - (IData)(1U)), 1U) 
                                              + (1U 
                                                 & (- (IData)(
                                                              VL_GTS_III(32, 0U, tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__srdhm_out)))))))))));
}

VL_ATTR_COLD void Vtb_bench___024root___eval_stl(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_stl\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VstlTriggered[0U])) {
        Vtb_bench___024root___stl_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD bool Vtb_bench___024root___eval_phase__stl(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___eval_phase__stl\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VstlExecute;
    // Body
    Vtb_bench___024root___eval_triggers_vec__stl(vlSelf);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_bench___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
    }
#endif
    __VstlExecute = Vtb_bench___024root___trigger_anySet__stl(vlSelfRef.__VstlTriggered);
    if (__VstlExecute) {
        Vtb_bench___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

bool Vtb_bench___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_bench___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(Vtb_bench___024root___trigger_anySet__act(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: @(posedge tb_bench.clk)\n");
    }
    if ((1U & (IData)((triggers[0U] >> 1U)))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 1 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
    if ((1U & (IData)((triggers[0U] >> 2U)))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 2 is active: @( tb_bench.done)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtb_bench___024root___ctor_var_reset(Vtb_bench___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_bench___024root___ctor_var_reset\n"); );
    Vtb_bench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->vlNamep);
    vlSelf->tb_bench__DOT__done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 893834345146744930ull);
    for (int __Vi0 = 0; __Vi0 < 784; ++__Vi0) {
        vlSelf->tb_bench__DOT__image_mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 1205708496835278306ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__a0_addr = VL_SCOPED_RAND_RESET_I(10, __VscopeHash, 346379123692587953ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a0_dout = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 17749203537095862082ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a1_addr = VL_SCOPED_RAND_RESET_I(13, __VscopeHash, 7965267773896280940ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a1_dout = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 4248088526348126954ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a2_addr = VL_SCOPED_RAND_RESET_I(13, __VscopeHash, 15739720990266407881ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a2_dout = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 15474154413082840116ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a3_addr = VL_SCOPED_RAND_RESET_I(11, __VscopeHash, 18118710230507820934ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a3_dout = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 7125862376274188198ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a4_addr = VL_SCOPED_RAND_RESET_I(11, __VscopeHash, 14098946150868655642ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a4_dout = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 9710598429169736790ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a5_addr = VL_SCOPED_RAND_RESET_I(11, __VscopeHash, 6502820692950170362ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a5_dout = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 16731683350955698700ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a6_addr = VL_SCOPED_RAND_RESET_I(9, __VscopeHash, 4929860072855716814ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a6_dout = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 3133844154968340532ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a7_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 17077651785657438566ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a7_dout = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 17708402044169669808ull);
    vlSelf->tb_bench__DOT__u_top__DOT__a8_dout = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 8388307985264263258ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l0_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1111859075662240694ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l0_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17637922742698808126ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l0_y_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1591777556686066835ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l1_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11258651014230200619ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l1_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15865611792608307037ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l1_y_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7641944020694912096ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l2_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11741607819334700316ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l2_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17618137047502758470ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l2_y_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17761988368463551192ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l3_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14282129580189971519ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l3_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13092416809095780702ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l3_y_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16597588258494056286ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l4_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16988243476355472907ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l4_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16326304823441841154ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l4_y_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7417544786170968338ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l5_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2727571462903451165ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l5_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14766243219122024381ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l5_y_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12794316284903139544ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l6_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11115417603456965173ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l6_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12314246871554713149ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l6_y_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2081809081666445979ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l7_start = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14435494163128672811ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l7_done = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8279781486847503351ull);
    vlSelf->tb_bench__DOT__u_top__DOT__l7_y_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15023397535865758294ull);
    vlSelf->tb_bench__DOT__u_top__DOT__stage = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 5540770389136174526ull);
    vlSelf->tb_bench__DOT__u_top__DOT__cstate = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 7324223076707922713ull);
    vlSelf->tb_bench__DOT__u_top__DOT__cnext = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 18303570788657141176ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_a0__DOT____Vlvbound_h8104ec03__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 784; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_a0__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6036050133674367079ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__u_a1__DOT____Vlvbound_hd9a27771__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 5408; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_a1__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 13307855205743770261ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__u_a2__DOT____Vlvbound_hd9a27771__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 5408; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_a2__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 5426631766717799307ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__u_a3__DOT____Vlvbound_ha38edf7c__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 1352; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_a3__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 10955809042116168038ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__u_a4__DOT____Vlvbound_h8ea625ea__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 1936; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_a4__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6053519230842891082ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__u_a5__DOT____Vlvbound_h8ea625ea__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 1936; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_a5__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 17628902632197973303ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__u_a6__DOT____Vlvbound_h99f3a4a9__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 400; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_a6__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 727544771372774597ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__u_a7__DOT____Vlvbound_hb1bb03c6__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_a7__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 2786379290919107956ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__u_a8__DOT____Vlvbound_h8978cab3__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_a8__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 5615249576385803851ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__w_addr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 12486774210583326639ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__w_byte = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 13375159587011990123ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__b_dout = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3169238773832625402ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__m0_dout = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13688707998723814657ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__sh_dout = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 5767053571496426577ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__oh = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 16177956254119260875ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__ow = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 3081385933346405185ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__oc = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 17912905018295989706ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__kh_i = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 12562138701221668530ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__kw_i = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 13360324896241604415ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__ic = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1491144515431518219ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__acc = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 2761306520722919507ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__state = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 12832328589019618414ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__next = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 2874522164466271432ull);
    for (int __Vi0 = 0; __Vi0 < 72; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_w_rom__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 14823266716315402828ull);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_b_rom__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 14351288365611494126ull);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_m0_rom__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4519204999811595803ull);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_sh_rom__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 17720202255925501805ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv1__DOT__u_requant__DOT__rdpot_out = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5323454950521335536ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_relu1__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 17781045912480087061ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_relu1__DOT__next = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 12272827897623027589ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_relu1__DOT__i = VL_SCOPED_RAND_RESET_I(13, __VscopeHash, 7571959016426532277ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool1__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 18135498230268141417ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool1__DOT__next = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 13741196495871450163ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool1__DOT__oh = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 17560195986192074153ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool1__DOT__ow = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 8653168454325237898ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool1__DOT__oc = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 10031612838184999861ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool1__DOT__kh_i = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3485440256836222029ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool1__DOT__kw_i = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3505435018152836817ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool1__DOT__best = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 12934342637766918913ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__w_addr = VL_SCOPED_RAND_RESET_I(11, __VscopeHash, 3961770282780377342ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__w_byte = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 8665768981118917052ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__b_dout = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 2719640564017840211ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__m0_dout = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11817834035397855032ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__sh_dout = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 15994355511202355852ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__oh = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 414284518507987243ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__ow = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 3533035826141633592ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__oc = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 3661575293020869556ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__kh_i = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 1513770261410250958ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__kw_i = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 16182516327242361152ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__ic = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 12332774332178913542ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__acc = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10244370819382163756ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__state = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 18328677443588040485ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__next = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 2393486640065923592ull);
    for (int __Vi0 = 0; __Vi0 < 1152; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_w_rom__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 13919992591453302069ull);
    }
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_b_rom__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 194151175474397942ull);
    }
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_m0_rom__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11260477315760707618ull);
    }
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_sh_rom__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 11327924479053338539ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__u_conv2__DOT__u_requant__DOT__rdpot_out = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10490838421404789753ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_relu2__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 8963203326090048089ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_relu2__DOT__next = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 16136210254577562930ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_relu2__DOT__i = VL_SCOPED_RAND_RESET_I(11, __VscopeHash, 16587412416930416975ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool2__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 15681536168692924681ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool2__DOT__next = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 9082308064932537258ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool2__DOT__oh = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 2913290197812171447ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool2__DOT__ow = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 6192097953537408571ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool2__DOT__oc = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 13557962860301949267ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool2__DOT__kh_i = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8844156890481518095ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool2__DOT__kw_i = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11919523867238207531ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_pool2__DOT__best = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 1673098821786931275ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__w_addr = VL_SCOPED_RAND_RESET_I(12, __VscopeHash, 16078617249282904451ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__w_byte = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 17037259079638406888ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__b_dout = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3228368374023252675ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__m0_dout = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 826633537118057786ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__sh_dout = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 7167370027529600794ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__m_i = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 9213621449138770205ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__k_i = VL_SCOPED_RAND_RESET_I(9, __VscopeHash, 7834656907555539492ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__acc = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10212634045893741443ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__state = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 15628926542077693239ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__next = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 13934418869251495632ull);
    for (int __Vi0 = 0; __Vi0 < 4000; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__u_w_rom__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 18410826204335583446ull);
    }
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__u_b_rom__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5306684475669237951ull);
    }
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__u_m0_rom__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10072683215665861848ull);
    }
    for (int __Vi0 = 0; __Vi0 < 10; ++__Vi0) {
        vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__u_sh_rom__DOT__mem[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 758207316695246186ull);
    }
    vlSelf->tb_bench__DOT__u_top__DOT__u_fc__DOT__u_requant__DOT__rdpot_out = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15789761098174299551ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_argmax__DOT__state = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 16416841800573534102ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_argmax__DOT__next = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 11658452949310408235ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_argmax__DOT__i = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 12703621438143679424ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_idx = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 13443726205244133782ull);
    vlSelf->tb_bench__DOT__u_top__DOT__u_argmax__DOT__best_val = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 3975358594853233018ull);
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VstlTriggered[__Vi0] = 0;
    }
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VactTriggered[__Vi0] = 0;
    }
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VactTriggeredAcc[__Vi0] = 0;
    }
    vlSelf->__Vtrigprevexpr___TOP__tb_bench__DOT__clk__0 = 0;
    vlSelf->__Vtrigprevexpr___TOP__tb_bench__DOT__done__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VnbaTriggered[__Vi0] = 0;
    }
    vlSelf->__Vi = 0;
}
