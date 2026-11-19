//! Activations quantized to 8 bits, in the layout the `dp4a` kernel wants.
//!
//! Groups of 32, one scale each, `scale = max|x| / 127` — the same rule the CPU backend's
//! `QuantizedX` uses, so a row computed here and a row computed there see the same integers. Alongside
//! the scale the kernel needs each group's *sum* of quants, because Q4_K is affine: the `dmin`
//! term multiplies the activation sum, not just the dot.

/// One activation vector, ready to be uploaded.
#[derive(Clone, Debug)]
pub struct Q8 {
    /// The quantized values, 32 per group, in order.
    pub qs: Vec<i8>,
    /// One scale per group; a group of zeros gets scale `0`.
    pub d: Vec<f32>,
    /// The sum of each group's quantized values, which is what the `dmin` term needs.
    pub s: Vec<f32>,
}

impl Q8 {
    /// Quantize `x` in groups of 32. A trailing partial group is ignored.
    pub fn quantize(x: &[f32]) -> Self {
        let groups = x.len() / 32;
        let mut qs = vec![0_i8; groups * 32];
        let mut d = vec![0.0_f32; groups];
        let mut s = vec![0.0_f32; groups];
        for (group, chunk) in x.as_chunks::<32>().0.iter().take(groups).enumerate() {
            let largest = chunk
                .iter()
                .fold(0.0_f32, |acc, value| acc.max(value.abs()));
            let scale = if largest == 0.0 { 0.0 } else { largest / 127.0 };
            d[group] = scale;
            if scale == 0.0 {
                continue;
            }
            let mut sum = 0_i32;
            for (slot, value) in qs[group * 32..][..32].iter_mut().zip(chunk) {
                let q = (value / scale).round().clamp(-127.0, 127.0) as i8;
                *slot = q;
                sum += i32::from(q);
            }
            s[group] = sum as f32;
        }
        Self { qs, d, s }
    }

    /// Groups in the vector.
    pub fn groups(&self) -> usize {
        self.d.len()
    }
}

/// The same three arrays, quantized on the device so an activation never leaves the card.
pub struct Q8Device {
    /// Quantized values, `32` per group.
    pub qs: crate::device::DeviceBuffer<i8>,
    /// One scale per group.
    pub d: crate::device::DeviceBuffer<f32>,
    /// One quant sum per group.
    pub s: crate::device::DeviceBuffer<f32>,
}

impl Q8Device {
    /// Allocate the arrays for up to `values` activation values, to be filled by [`Self::run`].
    pub fn allocate(
        device: &crate::device::Device,
        values: usize,
    ) -> Result<Self, vipera_errors::ViperaError> {
        Ok(Self {
            qs: device.alloc::<i8>(values)?,
            d: device.alloc::<f32>(values / 32)?,
            s: device.alloc::<f32>(values / 32)?,
        })
    }

    /// Quantize `x` on the device, allocating the three arrays.
    pub fn quantize(
        device: &crate::device::Device,
        x: &crate::device::DeviceBuffer<f32>,
        groups: usize,
    ) -> Result<Self, vipera_errors::ViperaError> {
        let qs = device.alloc::<i8>(groups * 32)?;
        let d = device.alloc::<f32>(groups)?;
        let s = device.alloc::<f32>(groups)?;
        let mut this = Self { qs, d, s };
        this.run(device, x, groups)?;
        Ok(this)
    }

    /// Re-quantize into an existing set of arrays, so a decode step does not allocate.
    pub fn run(
        &mut self,
        device: &crate::device::Device,
        x: &crate::device::DeviceBuffer<f32>,
        groups: usize,
    ) -> Result<(), vipera_errors::ViperaError> {
        device.launch_quantize(x, &mut self.qs, &mut self.d, &mut self.s, groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_group_gets_a_zero_scale_and_the_sum_matches_the_quants() {
        let mut x = vec![0.0_f32; 64];
        x[32..]
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = i as f32 - 16.0);
        let q8 = Q8::quantize(&x);
        assert_eq!(q8.groups(), 2);
        assert_eq!(q8.d[0], 0.0);
        assert_eq!(q8.s[0], 0.0);
        let sum: i32 = q8.qs[32..].iter().map(|q| i32::from(*q)).sum();
        assert_eq!(q8.s[1], sum as f32);
        assert!((q8.d[1] - 16.0 / 127.0).abs() < 1e-9);
    }
}
