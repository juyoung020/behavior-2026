// scenemap 링크: SCENEMAP_LIB_DIR 가 있으면 그 폴더의 libscenemap(.a/.so)을, 없으면 가짜 구현(src/sim/integ/scenemap_stub)을 빌드해 링크.
// 헤더: src/scene_graph/scenemap/include/scenemap.h(sm_detections) + src/sim/integ/scenemap_stub/sm_api.h(함수 선언, scenemap.h 로 옮겨질 예정).
fn main() {
    println!("cargo:rustc-check-cfg=cfg(scenemap_real)");
    println!("cargo:rerun-if-env-changed=SCENEMAP_LIB_DIR");
    println!("cargo:rerun-if-changed=../scenemap_stub/sm_stub.cpp");
    println!("cargo:rerun-if-changed=../scenemap_stub/sm_api.h");
    if let Ok(dir) = std::env::var("SCENEMAP_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=scenemap");
        // libscenemap.a 의 의존: Spark-DSG(저장, ~/.local 또는 SPARK_DSG_LIB_DIR) · zlib(best view PNG)
        let sdsg = std::env::var("SPARK_DSG_LIB_DIR")
            .unwrap_or_else(|_| format!("{}/.local/lib", std::env::var("HOME").unwrap_or_default()));
        println!("cargo:rerun-if-env-changed=SPARK_DSG_LIB_DIR");
        println!("cargo:rustc-link-search=native={sdsg}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{sdsg}");
        println!("cargo:rustc-link-lib=spark_dsg");
        println!("cargo:rustc-link-lib=z");
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-cfg=scenemap_real");
        return;
    }
    cc::Build::new()
        .cpp(true)
        .file("../scenemap_stub/sm_stub.cpp")
        .include("../scenemap_stub")
        .include("../../../scene_graph/scenemap/include")
        .flag_if_supported("-std=c++20")
        .opt_level(2)
        .compile("scenemap_stub");
}
