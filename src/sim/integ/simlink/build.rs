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
