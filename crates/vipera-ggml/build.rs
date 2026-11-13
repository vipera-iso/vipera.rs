//! Compiles ggml's K-quant sources for the ADR-001 prototype.
//!
//! No `cc` crate: EXC-001 scopes this exception to a C toolchain and nothing else, so the
//! build shells out to `cc`/`ar` and fails loudly if either is missing.
//!
//! The ggml checkout is not vendored yet. Point `VIPERA_GGML_SRC` at one (default
//! `/tmp/llamacpp`, where this engine's llama.cpp reference build lives). Vendoring is
//! packaging work for the promote step, not a prerequisite for taking the measurement.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// SIMD levels, most specific first. ggml's x86 kernels select these at compile time
/// (`__AVX512VNNI__`, `__AVX512F__`, `__AVX2__`), so the flags decide which kernel the
/// measurement actually exercises — which is why they are printed.
const LEVELS: &[(&[&str], &str)] = &[
    (
        &[
            "-mavx512f",
            "-mavx512vl",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vnni",
            "-mfma",
            "-mavx2",
            "-mf16c",
        ],
        "avx512vnni",
    ),
    (
        &[
            "-mavx512f",
            "-mavx512bw",
            "-mavx512dq",
            "-mfma",
            "-mavx2",
            "-mf16c",
        ],
        "avx512",
    ),
    (&["-mavx2", "-mfma", "-mf16c"], "avx2"),
];

fn main() {
    println!("cargo:rerun-if-env-changed=VIPERA_GGML_SRC");
    println!("cargo:rerun-if-changed=shim.c");

    let root = env::var_os("VIPERA_GGML_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/llamacpp"));
    let sources = [
        root.join("ggml/src/ggml-quants.c"),
        root.join("ggml/src/ggml-cpu/arch/x86/quants.c"),
    ];
    for source in &sources {
        if !source.exists() {
            panic!(
                "ADR-001 bind: {} not found. Set VIPERA_GGML_SRC to a llama.cpp checkout.",
                source.display()
            );
        }
    }

    let cpu = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let mut level = LEVELS[LEVELS.len() - 1];
    for (flags, name) in LEVELS {
        let needed = match *name {
            // /proc/cpuinfo spells it `avx512_vnni`; the compile-time macro is
            // `__AVX512VNNI__`.
            "avx512vnni" => "avx512_vnni",
            "avx512" => "avx512bw",
            _ => "avx2",
        };
        if cpu.contains(needed) {
            level = (flags, name);
            break;
        }
    }
    println!("cargo:warning=ADR-001 bind compiled for {}", level.1);

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let includes = [
        root.join("ggml/include"),
        root.join("ggml/src"),
        root.join("ggml/src/ggml-cpu"),
    ];
    let mut objects = Vec::new();

    let mut units: Vec<(PathBuf, &str)> = sources.iter().map(|p| (p.clone(), "c")).collect();
    units.push((PathBuf::from("shim.c"), "c"));
    for (source, language) in units {
        let stem = source
            .file_stem()
            .expect("file stem")
            .to_string_lossy()
            .into_owned();
        let object = out.join(format!("{stem}.o"));
        let mut cmd = Command::new("cc");
        cmd.args(["-c", "-O3", "-std=c11", "-fPIC", "-DNDEBUG"])
            .args(level.0)
            .arg("-o")
            .arg(&object)
            .arg(&source);
        for include in &includes {
            cmd.arg("-I").arg(include);
        }
        if language == "c" {
            cmd.arg("-Wno-unused-function");
        }
        run(&mut cmd, &source);
        objects.push(object);
    }

    let archive = out.join("libvipera_ggml_q4k.a");
    let mut ar = Command::new("ar");
    ar.arg("crs").arg(&archive).args(&objects);
    run(&mut ar, Path::new("ar"));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=vipera_ggml_q4k");
    println!("cargo:rustc-link-lib=m");
}

fn run(cmd: &mut Command, what: &Path) {
    let status = cmd
        .status()
        .unwrap_or_else(|error| panic!("ADR-001 bind: cannot run {:?}: {error}", cmd));
    assert!(
        status.success(),
        "ADR-001 bind: failed to build {}",
        what.display()
    );
}
