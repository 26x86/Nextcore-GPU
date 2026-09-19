//! Metal Driver Track M3 — host↔guest transport sketch (contract-only).
//!
//! Defines clean-room message shapes for buffer create/destroy/copy and
//! validates wire frames. Parsing a well-formed buffer/copy shape does **not**
//! allocate guest buffers, run a queue, or invent a Metal device. Queue-class
//! opcodes are rejected explicitly. Guest Metal remains unverified.

use crate::metal_acceptance::MetalAcceptanceReport;
use crate::metal_public_abi::MetalPublicAbi;

/// Machine-checkable marker for M3 transport sketch (research doc lock).
pub const M3_TRANSPORT_DOC_MARKER: &str = "M3_TRANSPORT:buffer-copy-shapes-reject";

/// ASCII `NCMT` — NextCore Metal Transport (clean-room; not Apple PVG).
pub const METAL_TRANSPORT_MAGIC: u32 = 0x4E_43_4D_54;
/// Sketch wire version. Unsupported versions reject.
pub const METAL_TRANSPORT_VERSION: u16 = 1;
/// Fixed header size in bytes (little-endian fields).
pub const METAL_TRANSPORT_HEADER_BYTES: usize = 16;
/// Contract max payload for the sketch (rejects larger frames).
pub const METAL_TRANSPORT_MAX_PAYLOAD: u32 = 4096;

/// Wire opcodes. Only buffer/copy class is shape-validated at M3.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalTransportOpcode {
    BufferCreate = 0x0001,
    BufferDestroy = 0x0002,
    BufferCopy = 0x0003,
    /// Queue submit — always rejected (no queue execution at M3).
    QueueSubmit = 0x0010,
    /// Fence signal — always rejected.
    FenceSignal = 0x0011,
}

impl MetalTransportOpcode {
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x0001 => Some(Self::BufferCreate),
            0x0002 => Some(Self::BufferDestroy),
            0x0003 => Some(Self::BufferCopy),
            0x0010 => Some(Self::QueueSubmit),
            0x0011 => Some(Self::FenceSignal),
            _ => None,
        }
    }

    pub fn is_buffer_copy_class(self) -> bool {
        matches!(
            self,
            Self::BufferCreate | Self::BufferDestroy | Self::BufferCopy
        )
    }

    pub fn is_queue_class(self) -> bool {
        matches!(self, Self::QueueSubmit | Self::FenceSignal)
    }
}

/// Fixed 16-byte little-endian header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalTransportHeader {
    pub magic: u32,
    pub version: u16,
    pub opcode: u16,
    pub payload_len: u32,
    pub flags: u32,
}

impl MetalTransportHeader {
    pub const fn payload_size_for(opcode: MetalTransportOpcode) -> u32 {
        match opcode {
            MetalTransportOpcode::BufferCreate => 16,
            MetalTransportOpcode::BufferDestroy => 8,
            MetalTransportOpcode::BufferCopy => 40,
            MetalTransportOpcode::QueueSubmit | MetalTransportOpcode::FenceSignal => 0,
        }
    }

    pub fn encode(self) -> [u8; METAL_TRANSPORT_HEADER_BYTES] {
        let mut out = [0u8; METAL_TRANSPORT_HEADER_BYTES];
        out[0..4].copy_from_slice(&self.magic.to_le_bytes());
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6..8].copy_from_slice(&self.opcode.to_le_bytes());
        out[8..12].copy_from_slice(&self.payload_len.to_le_bytes());
        out[12..16].copy_from_slice(&self.flags.to_le_bytes());
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, MetalTransportReject> {
        if bytes.len() < METAL_TRANSPORT_HEADER_BYTES {
            return Err(MetalTransportReject::FrameTooShort);
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let opcode = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        let payload_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        Ok(Self {
            magic,
            version,
            opcode,
            payload_len,
            flags,
        })
    }
}

/// Buffer create payload (16 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferCreateShape {
    pub size: u64,
    pub alignment: u32,
    pub flags: u32,
}

impl BufferCreateShape {
    pub fn encode(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&self.size.to_le_bytes());
        out[8..12].copy_from_slice(&self.alignment.to_le_bytes());
        out[12..16].copy_from_slice(&self.flags.to_le_bytes());
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, MetalTransportReject> {
        if bytes.len() != 16 {
            return Err(MetalTransportReject::PayloadLengthMismatch {
                expected: 16,
                actual: bytes.len(),
            });
        }
        Ok(Self {
            size: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            alignment: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            flags: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        })
    }

    pub fn validate(self) -> Result<(), MetalTransportReject> {
        if self.size == 0 {
            return Err(MetalTransportReject::ZeroSize);
        }
        if self.alignment != 0 && !self.alignment.is_power_of_two() {
            return Err(MetalTransportReject::InvalidAlignment);
        }
        if self.alignment != 0 && (self.size % u64::from(self.alignment)) != 0 {
            return Err(MetalTransportReject::SizeNotAligned);
        }
        // Reserved flag bits must stay clear in the sketch.
        if self.flags != 0 {
            return Err(MetalTransportReject::UnknownFlags(self.flags));
        }
        Ok(())
    }
}

/// Buffer destroy payload (8 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDestroyShape {
    pub handle: u64,
}

impl BufferDestroyShape {
    pub fn encode(self) -> [u8; 8] {
        self.handle.to_le_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, MetalTransportReject> {
        if bytes.len() != 8 {
            return Err(MetalTransportReject::PayloadLengthMismatch {
                expected: 8,
                actual: bytes.len(),
            });
        }
        Ok(Self {
            handle: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        })
    }

    pub fn validate(self) -> Result<(), MetalTransportReject> {
        if self.handle == 0 {
            return Err(MetalTransportReject::InvalidHandle(0));
        }
        Ok(())
    }
}

/// Buffer copy payload (40 bytes) — clean-room host↔guest copy request shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferCopyShape {
    pub src_handle: u64,
    pub dst_handle: u64,
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}

impl BufferCopyShape {
    pub fn encode(self) -> [u8; 40] {
        let mut out = [0u8; 40];
        out[0..8].copy_from_slice(&self.src_handle.to_le_bytes());
        out[8..16].copy_from_slice(&self.dst_handle.to_le_bytes());
        out[16..24].copy_from_slice(&self.src_offset.to_le_bytes());
        out[24..32].copy_from_slice(&self.dst_offset.to_le_bytes());
        out[32..40].copy_from_slice(&self.size.to_le_bytes());
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, MetalTransportReject> {
        if bytes.len() != 40 {
            return Err(MetalTransportReject::PayloadLengthMismatch {
                expected: 40,
                actual: bytes.len(),
            });
        }
        Ok(Self {
            src_handle: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            dst_handle: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            src_offset: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            dst_offset: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            size: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
        })
    }

    pub fn validate(self) -> Result<(), MetalTransportReject> {
        if self.size == 0 {
            return Err(MetalTransportReject::ZeroSize);
        }
        if self.src_handle == 0 {
            return Err(MetalTransportReject::InvalidHandle(0));
        }
        if self.dst_handle == 0 {
            return Err(MetalTransportReject::InvalidHandle(0));
        }
        let src_end = self
            .src_offset
            .checked_add(self.size)
            .ok_or(MetalTransportReject::SizeOverflow)?;
        let dst_end = self
            .dst_offset
            .checked_add(self.size)
            .ok_or(MetalTransportReject::SizeOverflow)?;
        // Keep ends referenced so overflow is the only range check at M3
        // (no backing store exists to bounds-check against).
        let _ = (src_end, dst_end);

        if self.src_handle == self.dst_handle {
            let src_lo = self.src_offset;
            let src_hi = src_end;
            let dst_lo = self.dst_offset;
            let dst_hi = dst_end;
            let overlap = src_lo < dst_hi && dst_lo < src_hi;
            if overlap {
                return Err(MetalTransportReject::SelfCopyOverlap);
            }
        }
        Ok(())
    }
}

/// Validated buffer/copy class shapes (contract-only; not executed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalTransportShape {
    BufferCreate(BufferCreateShape),
    BufferDestroy(BufferDestroyShape),
    BufferCopy(BufferCopyShape),
}

impl MetalTransportShape {
    pub fn opcode(self) -> MetalTransportOpcode {
        match self {
            Self::BufferCreate(_) => MetalTransportOpcode::BufferCreate,
            Self::BufferDestroy(_) => MetalTransportOpcode::BufferDestroy,
            Self::BufferCopy(_) => MetalTransportOpcode::BufferCopy,
        }
    }
}

/// Explicit transport reject paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalTransportReject {
    FrameTooShort,
    InvalidMagic { got: u32 },
    UnsupportedVersion { got: u16 },
    UnknownOpcode(u16),
    PayloadTooLarge { got: u32 },
    PayloadLengthMismatch { expected: usize, actual: usize },
    FrameLengthMismatch { expected: usize, actual: usize },
    ZeroSize,
    SizeOverflow,
    InvalidAlignment,
    SizeNotAligned,
    UnknownFlags(u32),
    InvalidHandle(u64),
    SelfCopyOverlap,
    /// Queue / fence opcodes must not run at M3.
    QueueExecutionForbidden(MetalTransportOpcode),
    /// Even a valid buffer/copy shape must not execute (no device / no queue).
    TransportNotExecutable,
    /// Design D10 / metal_verified still unmet.
    AcceptanceUnmet,
}

impl std::fmt::Display for MetalTransportReject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FrameTooShort => write!(f, "transport frame shorter than header"),
            Self::InvalidMagic { got } => write!(f, "invalid transport magic {got:#010x}"),
            Self::UnsupportedVersion { got } => {
                write!(f, "unsupported transport version {got}")
            }
            Self::UnknownOpcode(op) => write!(f, "unknown transport opcode {op:#06x}"),
            Self::PayloadTooLarge { got } => {
                write!(f, "payload too large ({got} > {METAL_TRANSPORT_MAX_PAYLOAD})")
            }
            Self::PayloadLengthMismatch { expected, actual } => {
                write!(f, "payload length mismatch: expected {expected}, got {actual}")
            }
            Self::FrameLengthMismatch { expected, actual } => {
                write!(f, "frame length mismatch: expected {expected}, got {actual}")
            }
            Self::ZeroSize => write!(f, "zero-size buffer/copy rejected"),
            Self::SizeOverflow => write!(f, "offset+size overflow"),
            Self::InvalidAlignment => write!(f, "alignment must be 0 or power-of-two"),
            Self::SizeNotAligned => write!(f, "size not aligned to requested alignment"),
            Self::UnknownFlags(flags) => write!(f, "unknown flags {flags:#010x}"),
            Self::InvalidHandle(h) => write!(f, "invalid handle {h}"),
            Self::SelfCopyOverlap => write!(f, "self-copy with overlapping ranges rejected"),
            Self::QueueExecutionForbidden(op) => {
                write!(f, "queue-class opcode {op:?} forbidden at M3")
            }
            Self::TransportNotExecutable => {
                write!(f, "transport sketch is not executable; no guest Metal device")
            }
            Self::AcceptanceUnmet => {
                write!(f, "metal acceptance unmet; metal_verified remains false")
            }
        }
    }
}

impl std::error::Error for MetalTransportReject {}

/// Host↔guest transport sketch handle. Validates shapes; never executes work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalTransport {
    acceptance: MetalAcceptanceReport,
}

impl MetalTransport {
    /// Honest M3 surface: unmet acceptance; shapes only.
    pub fn current() -> Self {
        Self {
            acceptance: MetalAcceptanceReport::unmet(),
        }
    }

    pub fn from_abi(abi: &MetalPublicAbi) -> Self {
        Self {
            acceptance: abi.acceptance().clone(),
        }
    }

    pub fn acceptance(&self) -> &MetalAcceptanceReport {
        &self.acceptance
    }

    /// Parse + validate a full wire frame into a buffer/copy shape.
    ///
    /// Queue-class opcodes reject here. Valid shapes are returned for contract
    /// tests only — call [`Self::execute_shape`] to confirm non-execution.
    pub fn parse_frame(&self, frame: &[u8]) -> Result<MetalTransportShape, MetalTransportReject> {
        if self.acceptance.metal_verified || self.acceptance.d10_complete() {
            return Err(MetalTransportReject::AcceptanceUnmet);
        }

        let header = MetalTransportHeader::decode(frame)?;
        if header.magic != METAL_TRANSPORT_MAGIC {
            return Err(MetalTransportReject::InvalidMagic { got: header.magic });
        }
        if header.version != METAL_TRANSPORT_VERSION {
            return Err(MetalTransportReject::UnsupportedVersion {
                got: header.version,
            });
        }
        if header.payload_len > METAL_TRANSPORT_MAX_PAYLOAD {
            return Err(MetalTransportReject::PayloadTooLarge {
                got: header.payload_len,
            });
        }

        let expected_len = METAL_TRANSPORT_HEADER_BYTES
            .checked_add(header.payload_len as usize)
            .ok_or(MetalTransportReject::SizeOverflow)?;
        if frame.len() != expected_len {
            return Err(MetalTransportReject::FrameLengthMismatch {
                expected: expected_len,
                actual: frame.len(),
            });
        }

        let opcode = MetalTransportOpcode::from_u16(header.opcode)
            .ok_or(MetalTransportReject::UnknownOpcode(header.opcode))?;

        if opcode.is_queue_class() {
            return Err(MetalTransportReject::QueueExecutionForbidden(opcode));
        }

        let expected_payload = MetalTransportHeader::payload_size_for(opcode) as usize;
        if header.payload_len as usize != expected_payload {
            return Err(MetalTransportReject::PayloadLengthMismatch {
                expected: expected_payload,
                actual: header.payload_len as usize,
            });
        }
        // Header flags reserved for buffer/copy class.
        if header.flags != 0 {
            return Err(MetalTransportReject::UnknownFlags(header.flags));
        }

        let payload = &frame[METAL_TRANSPORT_HEADER_BYTES..];
        let shape = match opcode {
            MetalTransportOpcode::BufferCreate => {
                let body = BufferCreateShape::decode(payload)?;
                body.validate()?;
                MetalTransportShape::BufferCreate(body)
            }
            MetalTransportOpcode::BufferDestroy => {
                let body = BufferDestroyShape::decode(payload)?;
                body.validate()?;
                MetalTransportShape::BufferDestroy(body)
            }
            MetalTransportOpcode::BufferCopy => {
                let body = BufferCopyShape::decode(payload)?;
                body.validate()?;
                MetalTransportShape::BufferCopy(body)
            }
            MetalTransportOpcode::QueueSubmit | MetalTransportOpcode::FenceSignal => {
                return Err(MetalTransportReject::QueueExecutionForbidden(opcode));
            }
        };
        Ok(shape)
    }

    /// Encode a validated shape into a wire frame (contract helper / fuzz seed).
    pub fn encode_shape(shape: MetalTransportShape) -> Vec<u8> {
        let opcode = shape.opcode();
        let payload: Vec<u8> = match shape {
            MetalTransportShape::BufferCreate(s) => s.encode().to_vec(),
            MetalTransportShape::BufferDestroy(s) => s.encode().to_vec(),
            MetalTransportShape::BufferCopy(s) => s.encode().to_vec(),
        };
        let header = MetalTransportHeader {
            magic: METAL_TRANSPORT_MAGIC,
            version: METAL_TRANSPORT_VERSION,
            opcode: opcode as u16,
            payload_len: payload.len() as u32,
            flags: 0,
        };
        let mut frame = header.encode().to_vec();
        frame.extend_from_slice(&payload);
        frame
    }

    /// Always rejects. Valid shapes still do not run (no device, no queue).
    pub fn execute_shape(
        &self,
        _shape: MetalTransportShape,
    ) -> Result<(), MetalTransportReject> {
        if self.acceptance.metal_verified || self.acceptance.d10_complete() {
            return Err(MetalTransportReject::AcceptanceUnmet);
        }
        Err(MetalTransportReject::TransportNotExecutable)
    }

    /// Parse then refuse execution — single entry for ingest paths.
    pub fn ingest(&self, frame: &[u8]) -> Result<(), MetalTransportReject> {
        let shape = self.parse_frame(frame)?;
        self.execute_shape(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_create() -> MetalTransportShape {
        MetalTransportShape::BufferCreate(BufferCreateShape {
            size: 256,
            alignment: 16,
            flags: 0,
        })
    }

    fn sample_copy() -> MetalTransportShape {
        MetalTransportShape::BufferCopy(BufferCopyShape {
            src_handle: 1,
            dst_handle: 2,
            src_offset: 0,
            dst_offset: 0,
            size: 64,
        })
    }

    #[test]
    fn well_formed_buffer_shapes_parse_but_do_not_execute() {
        let transport = MetalTransport::current();
        for shape in [
            sample_create(),
            MetalTransportShape::BufferDestroy(BufferDestroyShape { handle: 7 }),
            sample_copy(),
        ] {
            let frame = MetalTransport::encode_shape(shape);
            let parsed = transport.parse_frame(&frame).expect("shape must parse");
            assert_eq!(parsed, shape);
            let err = transport
                .execute_shape(parsed)
                .expect_err("must not execute");
            assert_eq!(err, MetalTransportReject::TransportNotExecutable);
            assert_eq!(
                transport.ingest(&frame),
                Err(MetalTransportReject::TransportNotExecutable)
            );
        }
    }

    #[test]
    fn queue_opcodes_are_forbidden() {
        let transport = MetalTransport::current();
        for op in [
            MetalTransportOpcode::QueueSubmit,
            MetalTransportOpcode::FenceSignal,
        ] {
            let header = MetalTransportHeader {
                magic: METAL_TRANSPORT_MAGIC,
                version: METAL_TRANSPORT_VERSION,
                opcode: op as u16,
                payload_len: 0,
                flags: 0,
            };
            let frame = header.encode().to_vec();
            let err = transport.parse_frame(&frame).expect_err("queue must reject");
            assert_eq!(err, MetalTransportReject::QueueExecutionForbidden(op));
        }
    }

    #[test]
    fn reject_paths_for_malformed_frames() {
        let transport = MetalTransport::current();

        assert_eq!(
            transport.parse_frame(&[0u8; 8]),
            Err(MetalTransportReject::FrameTooShort)
        );

        let mut bad_magic = MetalTransport::encode_shape(sample_create());
        bad_magic[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            transport.parse_frame(&bad_magic),
            Err(MetalTransportReject::InvalidMagic { .. })
        ));

        let mut bad_ver = MetalTransport::encode_shape(sample_create());
        bad_ver[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert_eq!(
            transport.parse_frame(&bad_ver),
            Err(MetalTransportReject::UnsupportedVersion { got: 99 })
        );

        let mut unknown = MetalTransport::encode_shape(sample_create());
        unknown[6..8].copy_from_slice(&0x00FFu16.to_le_bytes());
        assert_eq!(
            transport.parse_frame(&unknown),
            Err(MetalTransportReject::UnknownOpcode(0x00FF))
        );

        assert_eq!(
            BufferCreateShape {
                size: 0,
                alignment: 0,
                flags: 0
            }
            .validate(),
            Err(MetalTransportReject::ZeroSize)
        );

        assert_eq!(
            BufferCopyShape {
                src_handle: 1,
                dst_handle: 1,
                src_offset: 0,
                dst_offset: 32,
                size: 64,
            }
            .validate(),
            Err(MetalTransportReject::SelfCopyOverlap)
        );

        assert_eq!(
            BufferCopyShape {
                src_handle: 1,
                dst_handle: 2,
                src_offset: u64::MAX - 8,
                dst_offset: 0,
                size: 16,
            }
            .validate(),
            Err(MetalTransportReject::SizeOverflow)
        );
    }

    #[test]
    fn fuzz_reject_adversarial_frames() {
        let transport = MetalTransport::current();
        // Deterministic adversarial corpus (contract fuzz / reject).
        let seeds: &[&[u8]] = &[
            &[],
            &[0xff; 15],
            &[0xff; 16],
            &[0x54, 0x4d, 0x43, 0x4e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0x54, 0x4d, 0x43, 0x4e, 1, 0, 1, 0, 0xff, 0xff, 0, 0, 0, 0, 0, 0],
        ];
        for seed in seeds {
            let result = transport.ingest(seed);
            assert!(result.is_err(), "adversarial frame must reject: {seed:?}");
        }

        // Mutate a valid frame one byte at a time; every mutation must either
        // fail parse or fail execute (never Ok).
        let good = MetalTransport::encode_shape(sample_copy());
        for i in 0..good.len() {
            let mut mutant = good.clone();
            mutant[i] ^= 0x5A;
            assert!(
                transport.ingest(&mutant).is_err(),
                "mutant at byte {i} must reject"
            );
        }

        // Length fuzz: truncate / extend valid frames.
        for len in 0..=good.len() + 8 {
            let mut buf = good.clone();
            buf.resize(len, 0xAB);
            assert!(
                transport.ingest(&buf).is_err(),
                "length {len} must not execute"
            );
        }
    }

    #[test]
    fn transport_preserves_m2_abi_honesty() {
        let abi = MetalPublicAbi::current();
        let transport = MetalTransport::from_abi(&abi);
        assert!(!transport.acceptance().metal_verified);
        assert!(abi.capabilities().asserts_all_unsupported());
        let frame = MetalTransport::encode_shape(sample_create());
        assert_eq!(
            transport.ingest(&frame),
            Err(MetalTransportReject::TransportNotExecutable)
        );
    }
}
