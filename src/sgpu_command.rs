//! Existing SGPU list payload, shared with APLS; no new transport protocol.
//! All wire integers are little-endian and independent of host pointer width.
use thiserror::Error;

pub const SGPU_MAX_COMMANDS: usize = 1024;
pub const SGPU_MAX_COMMAND_BYTES: usize = 4 + SGPU_MAX_COMMANDS * 29;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SgpuWireError {
    #[error("malformed SGPU command payload")]
    Malformed,
    #[error("SGPU command count or byte limit exceeded")]
    Limit,
    #[error("SGPU snapshot range overflow or out of bounds")]
    Range,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SgpuCommand {
    CopyBuffer {
        src: u64,
        dst: u64,
        size: u64,
    },
    RenderClear {
        color: [f32; 4],
    },
    ComputeDispatch {
        kernel_id: u32,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    },
    Present {
        buffer: u64,
    },
}

impl SgpuCommand {
    pub fn tag(&self) -> u8 {
        match self {
            SgpuCommand::CopyBuffer { .. } => 1,
            SgpuCommand::RenderClear { .. } => 2,
            SgpuCommand::ComputeDispatch { .. } => 3,
            SgpuCommand::Present { .. } => 4,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![self.tag()];
        match self {
            SgpuCommand::CopyBuffer { src, dst, size } => {
                out.extend_from_slice(&src.to_le_bytes());
                out.extend_from_slice(&dst.to_le_bytes());
                out.extend_from_slice(&size.to_le_bytes());
            }
            SgpuCommand::RenderClear { color } => {
                for c in color {
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
            SgpuCommand::ComputeDispatch {
                kernel_id,
                grid,
                block,
            } => {
                out.extend_from_slice(&kernel_id.to_le_bytes());
                for v in [grid.0, grid.1, grid.2, block.0, block.1, block.2] {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            SgpuCommand::Present { buffer } => {
                out.extend_from_slice(&buffer.to_le_bytes());
            }
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, SgpuWireError> {
        if bytes.is_empty() {
            return Err(SgpuWireError::Malformed);
        }
        match bytes[0] {
            1 => {
                if bytes.len() != 25 {
                    return Err(SgpuWireError::Malformed);
                }
                Ok(SgpuCommand::CopyBuffer {
                    src: u64::from_le_bytes(bytes[1..9].try_into().unwrap()),
                    dst: u64::from_le_bytes(bytes[9..17].try_into().unwrap()),
                    size: u64::from_le_bytes(bytes[17..25].try_into().unwrap()),
                })
            }
            2 => {
                if bytes.len() != 17 {
                    return Err(SgpuWireError::Malformed);
                }
                let mut color = [0f32; 4];
                for (i, c) in color.iter_mut().enumerate() {
                    *c = f32::from_le_bytes(bytes[1 + i * 4..5 + i * 4].try_into().unwrap());
                }
                Ok(SgpuCommand::RenderClear { color })
            }
            3 => {
                if bytes.len() != 29 {
                    return Err(SgpuWireError::Malformed);
                }
                let kernel_id = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
                let vals = (0..6)
                    .map(|i| u32::from_le_bytes(bytes[5 + i * 4..9 + i * 4].try_into().unwrap()))
                    .collect::<Vec<_>>();
                Ok(SgpuCommand::ComputeDispatch {
                    kernel_id,
                    grid: (vals[0], vals[1], vals[2]),
                    block: (vals[3], vals[4], vals[5]),
                })
            }
            4 => {
                if bytes.len() != 9 {
                    return Err(SgpuWireError::Malformed);
                }
                Ok(SgpuCommand::Present {
                    buffer: u64::from_le_bytes(bytes[1..9].try_into().unwrap()),
                })
            }
            _ => Err(SgpuWireError::Malformed),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SgpuCommandList {
    pub cmds: Vec<SgpuCommand>,
}

impl SgpuCommandList {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = (self.cmds.len() as u32).to_le_bytes().to_vec();
        for c in &self.cmds {
            out.extend_from_slice(&c.encode());
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, SgpuWireError> {
        if bytes.len() > SGPU_MAX_COMMAND_BYTES {
            return Err(SgpuWireError::Limit);
        }
        if bytes.len() < 4 {
            return Err(SgpuWireError::Malformed);
        }
        let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        if count > SGPU_MAX_COMMANDS {
            return Err(SgpuWireError::Limit);
        }
        // Every known command is at least nine bytes. Reject attacker-controlled
        // counts before reserving a Vec, even when the transfer is truncated.
        if count > (bytes.len() - 4) / 9 {
            return Err(SgpuWireError::Malformed);
        }
        let mut cursor = 4usize;
        let mut cmds = Vec::with_capacity(count);
        for _ in 0..count {
            let length = match bytes.get(cursor) {
                Some(1) => 25,
                Some(2) => 17,
                Some(3) => 29,
                Some(4) => 9,
                _ => return Err(SgpuWireError::Malformed),
            };
            let end = cursor.checked_add(length).ok_or(SgpuWireError::Range)?;
            let record = bytes.get(cursor..end).ok_or(SgpuWireError::Malformed)?;
            cmds.push(SgpuCommand::decode(record)?);
            cursor = end;
        }
        if cursor != bytes.len() {
            return Err(SgpuWireError::Malformed);
        }
        Ok(Self { cmds })
    }

    /// Decode copied command values from a caller-validated immutable snapshot.
    /// Offsets are byte indices, never host pointers or dereferenced guest VAs.
    pub fn decode_window(snapshot: &[u8], offset: u64, length: u64) -> Result<Self, SgpuWireError> {
        if length > SGPU_MAX_COMMAND_BYTES as u64 {
            return Err(SgpuWireError::Limit);
        }
        let end = offset.checked_add(length).ok_or(SgpuWireError::Range)?;
        let start = usize::try_from(offset).map_err(|_| SgpuWireError::Range)?;
        let end = usize::try_from(end).map_err(|_| SgpuWireError::Range)?;
        Self::decode(snapshot.get(start..end).ok_or(SgpuWireError::Range)?)
    }
}
