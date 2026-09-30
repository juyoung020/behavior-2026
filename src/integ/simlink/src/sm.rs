//! scenemap C ABI(src/integ/scenemap_stub/sm_api.h, 같은 프로세스) FFI + 안전 래퍼 + 계획기 질의([`SceneQuery`]).
//! 링크 대상은 build.rs 가 고른다: libscenemap(SCENEMAP_LIB_DIR) 또는 가짜 구현(sm_stub.cpp).

use bagent::graph::{Grid2, ObjState, SceneObject, SceneQuery};
use bagent::odom::Pose;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::Arc;

#[repr(C)]
pub struct SmProprio {
    pub stamp: f64,
    pub proprio: *const f32,
    pub n_proprio: c_int,
}

#[repr(C)]
pub struct SmImage {
    pub stamp: f64,
    pub cam: c_int,
    pub w: c_int,
    pub h: c_int,
    pub rgba: *const u8,
    pub depth_m: *const f32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct SmPose2 {
    pub stamp: f64,
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SmObject {
    pub id: u32,
    pub name: *const c_char,
    pub score: f32,
    pub pos: [f64; 3],
    pub extent: [f64; 3],
    pub first_pos: [f64; 3],
    pub n_obs: u32,
    pub last_seen: f64,
    pub state: i32,
    pub handled: i32,
    pub structural: i32,
}

#[repr(C)]
pub struct SmGrid {
    pub resolution: f64,
    pub origin: [f64; 2],
    pub width: i32,
    pub height: i32,
    pub cells: *const i8,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct SmStatus {
    pub last_proprio_stamp: f64,
    pub last_image_stamp: f64,
    pub n_objects: i32,
    pub n_images: i32,
    pub n_proprio: i32,
}

#[repr(C)]
pub struct SmCtx {
    _p: [u8; 0],
}
#[repr(C)]
pub struct SmSnap {
    _p: [u8; 0],
}

extern "C" {
    fn sm_create(config_json: *const c_char) -> *mut SmCtx;
    fn sm_destroy(c: *mut SmCtx);
    fn sm_set_labels(c: *mut SmCtx, names: *const *const c_char, n: c_int) -> c_int;
    fn sm_reset(c: *mut SmCtx) -> c_int;
    fn sm_push_proprio(c: *mut SmCtx, p: *const SmProprio) -> c_int;
    fn sm_push_image(c: *mut SmCtx, im: *const SmImage, dets: *const c_void) -> c_int;
    fn sm_mark_handled(c: *mut SmCtx, id: u32) -> c_int;
    fn sm_snapshot(c: *mut SmCtx, out: *mut *mut SmSnap) -> c_int;
    fn sm_snapshot_release(s: *mut SmSnap);
    fn sm_snap_pose(s: *const SmSnap) -> SmPose2;
    fn sm_snap_status(s: *const SmSnap) -> SmStatus;
    fn sm_snap_objects(s: *const SmSnap, out: *mut *const SmObject) -> c_int;
    fn sm_snap_find(s: *const SmSnap, name: *const c_char, ids: *mut u32, scores: *mut f32, cap: c_int) -> c_int;
    fn sm_snap_near(s: *const SmSnap, p: *const f64, r: f64, ids: *mut u32, cap: c_int) -> c_int;
    fn sm_snap_map(s: *const SmSnap, out: *mut SmGrid) -> c_int;
    fn sm_snap_reachable(s: *const SmSnap, from: *const f64, to: *const f64) -> f64;
}

#[cfg(not(scenemap_real))]
#[allow(dead_code)]
extern "C" {
    fn sm_stub_add_object(c: *mut SmCtx, id: u32, name: *const c_char, x: f64, y: f64, z: f64) -> c_int;
}

/// scenemap 인스턴스 하나. 입력(push)은 관측 스레드 하나에서, 질의(snapshot)·표시는 아무 스레드에서(sm_api.h 약속).
pub struct Scenemap {
    ctx: *mut SmCtx,
}
unsafe impl Send for Scenemap {}
unsafe impl Sync for Scenemap {}

impl Drop for Scenemap {
    fn drop(&mut self) {
        unsafe { sm_destroy(self.ctx) }
    }
}

fn rc(r: c_int, what: &str) -> Result<(), String> {
    if r < 0 {
        Err(format!("scenemap {what}: {r}"))
    } else {
        Ok(())
    }
}

/// 스냅숏(읽기 전용, 떨어지면 놓음)
pub struct Snap(*mut SmSnap);
impl Drop for Snap {
    fn drop(&mut self) {
        unsafe { sm_snapshot_release(self.0) }
    }
}

impl Snap {
    pub fn pose(&self) -> SmPose2 {
        unsafe { sm_snap_pose(self.0) }
    }
    pub fn status(&self) -> SmStatus {
        unsafe { sm_snap_status(self.0) }
    }
    pub fn objects(&self) -> Vec<SceneObject> {
        let mut p: *const SmObject = std::ptr::null();
        let n = unsafe { sm_snap_objects(self.0, &mut p) };
        if n <= 0 || p.is_null() {
            return vec![];
        }
        let s = unsafe { std::slice::from_raw_parts(p, n as usize) };
        s.iter()
            .map(|o| SceneObject {
                id: o.id,
                name: if o.name.is_null() { String::new() } else { unsafe { CStr::from_ptr(o.name) }.to_string_lossy().into_owned() },
                score: o.score as f64,
                position: o.pos,
                extent: o.extent,
                first_position: o.first_pos,
                n_obs: o.n_obs,
                last_seen: o.last_seen,
                state: match o.state {
                    1 => ObjState::Gone,
                    2 => ObjState::Moved,
                    3 => ObjState::Held,
                    _ => ObjState::Seen,
                },
                handled: o.handled != 0,
                structural: o.structural != 0,
                room: None,
            })
            .collect()
    }
    fn ids_to_objs(&self, ids: &[u32], scores: Option<&[f32]>) -> Vec<SceneObject> {
        let all = self.objects();
        ids.iter()
            .enumerate()
            .filter_map(|(k, id)| {
                all.iter().find(|o| o.id == *id).cloned().map(|mut o| {
                    if let Some(s) = scores {
                        o.score = s[k] as f64;
                    }
                    o
                })
            })
            .collect()
    }
}

impl Scenemap {
    pub fn new(config_json: &str) -> Result<Scenemap, String> {
        let c = CString::new(config_json).map_err(|e| e.to_string())?;
        let ctx = unsafe { sm_create(if config_json.is_empty() { std::ptr::null() } else { c.as_ptr() }) };
        if ctx.is_null() {
            return Err("sm_create 실패".into());
        }
        Ok(Scenemap { ctx })
    }
    pub fn is_stub() -> bool {
        !cfg!(scenemap_real)
    }
    pub fn reset(&self) -> Result<(), String> {
        rc(unsafe { sm_reset(self.ctx) }, "reset")
    }
    pub fn set_labels(&self, labels: &[String]) -> Result<(), String> {
        let cs: Vec<CString> = labels.iter().map(|l| CString::new(l.as_str()).unwrap_or_default()).collect();
        let ps: Vec<*const c_char> = cs.iter().map(|c| c.as_ptr()).collect();
        rc(unsafe { sm_set_labels(self.ctx, ps.as_ptr(), ps.len() as c_int) }, "set_labels")
    }
    pub fn push_proprio(&self, stamp: f64, p: &[f32]) -> Result<(), String> {
        let a = SmProprio { stamp, proprio: p.as_ptr(), n_proprio: p.len() as c_int };
        rc(unsafe { sm_push_proprio(self.ctx, &a) }, "push_proprio")
    }
    /// 영상 한 장(RGBA·깊이 중 있는 것). dets = NULL → scenemap 이 검출기(YOLOE)를 부른다.
    #[allow(clippy::too_many_arguments)]
    pub fn push_image(&self, stamp: f64, cam: u8, w: u32, h: u32, rgba: Option<&[u8]>, depth: Option<&[u8]>, k: [f64; 4]) -> Result<(), String> {
        let im = SmImage {
            stamp,
            cam: cam as c_int,
            w: w as c_int,
            h: h as c_int,
            rgba: rgba.map(|r| r.as_ptr()).unwrap_or(std::ptr::null()),
            depth_m: depth.map(|d| d.as_ptr() as *const f32).unwrap_or(std::ptr::null()),
            fx: k[0],
            fy: k[1],
            cx: k[2],
            cy: k[3],
        };
        rc(unsafe { sm_push_image(self.ctx, &im, std::ptr::null()) }, "push_image")
    }
    pub fn mark_handled(&self, id: u32) -> Result<(), String> {
        rc(unsafe { sm_mark_handled(self.ctx, id) }, "mark_handled")
    }
    pub fn snapshot(&self) -> Result<Snap, String> {
        let mut p: *mut SmSnap = std::ptr::null_mut();
        rc(unsafe { sm_snapshot(self.ctx, &mut p) }, "snapshot")?;
        if p.is_null() {
            return Err("scenemap snapshot NULL".into());
        }
        Ok(Snap(p))
    }
    /// 가짜 구현 전용: 시험에서 물체 넣기
    #[cfg(not(scenemap_real))]
    #[allow(dead_code)]
    pub fn stub_add_object(&self, id: u32, name: &str, p: [f64; 3]) {
        let c = CString::new(name).unwrap_or_default();
        unsafe {
            sm_stub_add_object(self.ctx, id, c.as_ptr(), p[0], p[1], p[2]);
        }
    }
}

/// 계획기 질의 = scenemap 스냅숏.
pub struct SmQuery(pub Arc<Scenemap>);

const CAP: usize = 256;

impl SceneQuery for SmQuery {
    fn objects(&self) -> Result<Vec<SceneObject>, String> {
        Ok(self.0.snapshot()?.objects())
    }
    fn near(&self, p: [f64; 3], r: f64) -> Result<Vec<SceneObject>, String> {
        let s = self.0.snapshot()?;
        let mut ids = vec![0u32; CAP];
        let n = unsafe { sm_snap_near(s.0, p.as_ptr(), r, ids.as_mut_ptr(), CAP as c_int) }.max(0) as usize;
        Ok(s.ids_to_objs(&ids[..n.min(CAP)], None))
    }
    fn find(&self, name: &str) -> Result<Vec<SceneObject>, String> {
        let s = self.0.snapshot()?;
        let c = CString::new(name).map_err(|e| e.to_string())?;
        let mut ids = vec![0u32; CAP];
        let mut sc = vec![0f32; CAP];
        let n = unsafe { sm_snap_find(s.0, c.as_ptr(), ids.as_mut_ptr(), sc.as_mut_ptr(), CAP as c_int) }.max(0) as usize;
        let n = n.min(CAP);
        Ok(s.ids_to_objs(&ids[..n], Some(&sc[..n])))
    }
    fn map(&self) -> Result<Grid2, String> {
        let s = self.0.snapshot()?;
        let mut g = SmGrid { resolution: 0.0, origin: [0.0; 2], width: 0, height: 0, cells: std::ptr::null() };
        rc(unsafe { sm_snap_map(s.0, &mut g) }, "map")?;
        let n = (g.width.max(0) as usize) * (g.height.max(0) as usize);
        let cells = if n > 0 && !g.cells.is_null() { unsafe { std::slice::from_raw_parts(g.cells, n) }.to_vec() } else { vec![] };
        Ok(Grid2 { resolution: g.resolution, origin: g.origin, width: g.width.max(0) as u32, height: g.height.max(0) as u32, cells })
    }
    fn reachable(&self, from: [f64; 2], to: [f64; 2]) -> Result<Option<f64>, String> {
        let s = self.0.snapshot()?;
        let d = unsafe { sm_snap_reachable(s.0, from.as_ptr(), to.as_ptr()) };
        Ok((d >= 0.0).then_some(d))
    }
    fn pose(&self) -> Result<(f64, Pose), String> {
        let p = self.0.snapshot()?.pose();
        Ok((p.stamp, Pose { x: p.x, y: p.y, yaw: p.yaw }))
    }
    fn mark_handled(&self, id: u32) -> Result<(), String> {
        self.0.mark_handled(id)
    }
    fn set_labels(&self, labels: &[String]) -> Result<(), String> {
        self.0.set_labels(labels)
    }
    fn describe(&self) -> String {
        if Scenemap::is_stub() { "stub".into() } else { "libscenemap".into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_roundtrip_with_stub() {
        let sm = Arc::new(Scenemap::new("").unwrap());
        sm.set_labels(&["radio receiver".into(), "coffee table".into()]).unwrap();
        // 1 m/s 로 31 스텝 → 1 m (프레임 i 는 i-1 속도)
        let mut p = vec![0f32; 61];
        p[0] = 1.0;
        for k in 0..31 {
            sm.push_proprio(k as f64 / 30.0, &p).unwrap();
        }
        let q = SmQuery(sm.clone());
        let (t, pose) = q.pose().unwrap();
        assert!((t - 1.0).abs() < 1e-9);
        assert!((pose.x - 1.0).abs() < 1e-6 && pose.y.abs() < 1e-9);
        let rgba = vec![1u8; 8 * 8 * 4];
        let depth: Vec<u8> = (0..64).flat_map(|_| 1.5f32.to_le_bytes()).collect();
        sm.push_image(29.0 / 30.0, 0, 8, 8, Some(&rgba), Some(&depth), [3.4, 3.4, 4.0, 4.0]).unwrap();
        let st = sm.snapshot().unwrap().status();
        assert_eq!(st.n_images, 1);
        assert!((st.last_image_stamp - 29.0 / 30.0).abs() < 1e-12);
        if Scenemap::is_stub() {
            #[cfg(not(scenemap_real))]
            {
                sm.stub_add_object(3, "radio receiver", [1.0, 0.5, 0.6]);
                sm.stub_add_object(7, "coffee table", [1.2, 0.4, 0.3]);
            }
            let f = q.find("radio").unwrap();
            assert_eq!(f.len(), 1);
            assert_eq!((f[0].id, f[0].name.as_str()), (3, "radio receiver"));
            let n = q.near([1.0, 0.5, 0.5], 0.5).unwrap();
            assert_eq!(n.iter().map(|o| o.id).collect::<Vec<_>>(), vec![3, 7]);
            q.mark_handled(7).unwrap();
            assert!(q.objects().unwrap().iter().find(|o| o.id == 7).unwrap().handled);
            // 계획기 SceneGraph 자리에서
            let mut g = bagent::graph::ScenemapGraph { q: Arc::new(SmQuery(sm.clone())) };
            use bagent::graph::SceneGraph;
            assert_eq!(g.query("coffee table", 5).unwrap()[0].id, "7");
        }
        sm.reset().unwrap();
        assert_eq!(sm.snapshot().unwrap().status().n_proprio, 0);
    }
}
