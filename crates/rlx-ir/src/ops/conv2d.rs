// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! NCHW convolution builders (`conv2d`, `conv_transpose2d`).

use crate::{Graph, NodeId, Op};

impl Graph {
    /// 2D convolution on NCHW tensors (`Op::Conv`). Weight `[C_out, C_in/g, kH, kW]`.
    pub fn conv2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> NodeId {
        let in_s = self.node(input).shape.clone();
        let w_s = self.node(weight).shape.clone();
        let out = crate::shape::conv2d_output_shape(
            &in_s,
            &w_s,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        )
        .expect("conv2d shape inference");
        self.push(
            Op::Conv {
                kernel_size: kernel_size.to_vec(),
                stride: stride.to_vec(),
                padding: padding.to_vec(),
                dilation: dilation.to_vec(),
                groups,
            },
            vec![input, weight],
            out,
            None,
        )
    }

    /// NCHW im2col (`Op::Im2Col`). Output `[M, C·kH·kW]`.
    pub fn im2col(
        &mut self,
        input: NodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> NodeId {
        let in_s = self.node(input).shape.clone();
        let out = crate::shape::im2col_output_shape(&in_s, kernel_size, stride, padding, dilation)
            .expect("im2col shape inference");
        self.push(
            Op::Im2Col {
                kernel_size: kernel_size.to_vec(),
                stride: stride.to_vec(),
                padding: padding.to_vec(),
                dilation: dilation.to_vec(),
            },
            vec![input],
            out,
            None,
        )
    }

    /// 2D transposed convolution on NCHW. Weight `[C_in, C_out/g, kH, kW]`.
    pub fn conv_transpose2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_padding: [usize; 2],
        groups: usize,
    ) -> NodeId {
        let in_s = self.node(input).shape.clone();
        let w_s = self.node(weight).shape.clone();
        let out = crate::shape::conv_transpose2d_output_shape(
            &in_s,
            &w_s,
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
        )
        .expect("conv_transpose2d shape inference");
        self.push(
            Op::ConvTranspose2d {
                kernel_size: kernel_size.to_vec(),
                stride: stride.to_vec(),
                padding: padding.to_vec(),
                dilation: dilation.to_vec(),
                output_padding: output_padding.to_vec(),
                groups,
            },
            vec![input, weight],
            out,
            None,
        )
    }
}
