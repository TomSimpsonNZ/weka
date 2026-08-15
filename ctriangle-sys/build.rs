use std::path::PathBuf;

fn main() {
    // The original C/C++ source lives in ../../triangle relative to this crate.
    let triangle_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("triangle");
    let source = triangle_dir.join("triangle.cpp");

    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-changed={}", triangle_dir.join("triangle.h").display());

    let shim = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shim.cpp");
    println!("cargo:rerun-if-changed={}", shim.display());

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file(&source)
        .file(&shim)
        .include(&triangle_dir)
        .define("TRILIBRARY", None)
        .define("ANSI_DECLARATORS", None)
        .define("NO_TIMER", None)
        // Triangle is legacy code; silence its many warnings so they don't drown the build.
        .warnings(false)
        .opt_level(2);

    // x86 needs the FPU register incantations; Apple Silicon / aarch64 does not.
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "x86" || target_arch == "x86_64" {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_os == "linux" {
            build.define("LINUX", None);
        }
    }

    build.compile("triangle_c");
}
