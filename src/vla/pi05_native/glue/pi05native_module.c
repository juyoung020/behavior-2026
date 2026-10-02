/* _pi05native: thin CPython extension over the C API (raw C API + buffer protocol; no pybind11, no numpy
 * headers, no torch). Arrays come in through the buffer protocol (numpy arrays, CPU torch tensors via .numpy()),
 * CUDA images as raw device pointers (from __cuda_array_interface__). Outputs are written into caller-provided
 * writable buffers. The GIL is released while the engine runs. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "../include/pi05_native.h"

static const char* CAP = "pi05native.engine";

static Pi05Engine* get_engine(PyObject* cap) { return (Pi05Engine*)PyCapsule_GetPointer(cap, CAP); }

static void cap_free(PyObject* cap) {
  Pi05Engine* e = (Pi05Engine*)PyCapsule_GetPointer(cap, CAP);
  if (e) pi05_destroy(e);
}

static PyObject* py_create(PyObject* self, PyObject* args) {
  const char* path;
  int device = 0;
  if (!PyArg_ParseTuple(args, "s|i", &path, &device)) return NULL;
  char err[512] = {0};
  Pi05Engine* e;
  Py_BEGIN_ALLOW_THREADS e = pi05_create(path, device, err, sizeof err);
  Py_END_ALLOW_THREADS if (!e) {
    PyErr_SetString(PyExc_RuntimeError, err);
    return NULL;
  }
  return PyCapsule_New(e, CAP, cap_free);
}

static PyObject* py_info(PyObject* self, PyObject* args) {
  PyObject* cap;
  if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
  Pi05Engine* e = get_engine(cap);
  if (!e) return NULL;
  Pi05Info i;
  pi05_info(e, &i);
  return Py_BuildValue("{s:i,s:i,s:i,s:L,s:L,s:s,s:(sss),s:i,s:i}", "action_horizon", i.action_horizon, "action_dim",
                       i.action_dim, "proprio_min_len", i.proprio_min_len, "weight_bytes", (long long)i.weight_bytes,
                       "activation_bytes", (long long)i.activation_bytes, "robot_name", i.robot_name, "cam_keys",
                       i.cam_keys[0], i.cam_keys[1], i.cam_keys[2], "model_kind", i.model_kind, "num_steps",
                       i.num_steps);
}

/* image argument: a buffer (H, W, C>=3) uint8, or a tuple (device_ptr, h, w, row_stride, pix_stride) */
static int get_image(PyObject* o, Pi05Image* im, Py_buffer* view, int* has_view) {
  *has_view = 0;
  if (PyTuple_Check(o)) {
    unsigned long long ptr;
    int h, w;
    long long rs, ps;
    if (!PyArg_ParseTuple(o, "KiiLL", &ptr, &h, &w, &rs, &ps)) return -1;
    im->data = (const uint8_t*)(uintptr_t)ptr;
    im->h = h;
    im->w = w;
    im->row_stride = rs;
    im->pix_stride = ps;
    im->on_device = 1;
    return 0;
  }
  if (PyObject_GetBuffer(o, view, PyBUF_STRIDED_RO | PyBUF_FORMAT) != 0) return -1;
  *has_view = 1;
  if (view->itemsize != 1 || (view->ndim != 3 && view->ndim != 4) || view->shape[view->ndim - 1] < 3 ||
      view->strides[view->ndim - 1] != 1) {
    PyErr_SetString(PyExc_ValueError, "image must be uint8 [H, W, C>=3] (or [1, H, W, C]) with contiguous channels");
    return -1;
  }
  int o0 = view->ndim - 3;
  if (o0 == 1 && view->shape[0] != 1) {
    PyErr_SetString(PyExc_ValueError, "4-d image must have batch 1");
    return -1;
  }
  im->data = (const uint8_t*)view->buf;
  im->h = (int)view->shape[o0];
  im->w = (int)view->shape[o0 + 1];
  im->row_stride = view->strides[o0];
  im->pix_stride = view->strides[o0 + 1];
  im->on_device = 0;
  return 0;
}

static int get_f32(PyObject* o, Py_buffer* v, const char* what) {
  if (PyObject_GetBuffer(o, v, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) return -1;
  if (v->itemsize != 4 || !v->format || (v->format[0] != 'f' && !(v->format[0] == '<' && v->format[1] == 'f') &&
                                         !(v->format[0] == '=' && v->format[1] == 'f'))) {
    PyErr_Format(PyExc_ValueError, "%s must be float32", what);
    PyBuffer_Release(v);
    return -1;
  }
  return 0;
}

static PyObject* timing_tuple(const Pi05Timing* t) {
  return Py_BuildValue("{s:f,s:f,s:f,s:f,s:f,s:i,s:i}", "preprocess_ms", t->preprocess_ms, "upload_ms", t->upload_ms,
                       "gpu_ms", t->gpu_ms, "postprocess_ms", t->postprocess_ms, "total_ms", t->total_ms, "tokens",
                       t->tokens, "truncated", t->truncated);
}

/* infer(engine, img0, img1, img2, proprio_f32, prompt, noise_f32_or_None, out_f64[ah*ad]) -> timing dict */
static PyObject* py_infer(PyObject* self, PyObject* args) {
  PyObject *cap, *im[3], *prop, *noise, *out;
  const char* prompt;
  if (!PyArg_ParseTuple(args, "OOOOOsOO", &cap, &im[0], &im[1], &im[2], &prop, &prompt, &noise, &out)) return NULL;
  Pi05Engine* e = get_engine(cap);
  if (!e) return NULL;
  Pi05Image imgs[3];
  Py_buffer iv[3], pv, nv, ov;
  int has[3] = {0, 0, 0}, hn = 0, ok = 0;
  PyObject* ret = NULL;
  for (int i = 0; i < 3; ++i)
    if (get_image(im[i], &imgs[i], &iv[i], &has[i]) != 0) goto done_img;
  if (get_f32(prop, &pv, "proprio") != 0) goto done_img;
  if (noise != Py_None) {
    if (get_f32(noise, &nv, "noise") != 0) goto done_prop;
    hn = 1;
  }
  if (PyObject_GetBuffer(out, &ov, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) != 0) goto done_noise;
  {
    Pi05Info info;
    pi05_info(e, &info);
    if (ov.len < (Py_ssize_t)(info.action_horizon * info.action_dim * 8)) {
      PyErr_SetString(PyExc_ValueError, "out buffer too small (float64 [action_horizon, action_dim])");
      goto done_out;
    }
    Pi05Timing t;
    int rc;
    Py_BEGIN_ALLOW_THREADS rc = pi05_infer(e, imgs, (const float*)pv.buf, (int)(pv.len / 4), prompt,
                                           hn ? (const float*)nv.buf : NULL, (double*)ov.buf, &t);
    Py_END_ALLOW_THREADS if (rc != 0) {
      PyErr_SetString(PyExc_RuntimeError, pi05_last_error(e));
      goto done_out;
    }
    ret = timing_tuple(&t);
    ok = 1;
  }
done_out:
  PyBuffer_Release(&ov);
done_noise:
  if (hn) PyBuffer_Release(&nv);
done_prop:
  PyBuffer_Release(&pv);
done_img:
  for (int i = 0; i < 3; ++i)
    if (has[i]) PyBuffer_Release(&iv[i]);
  (void)ok;
  return ret;
}

/* act(engine, slot, img0, img1, img2, proprio_f32, prompt, replan_every, out_f32[ad]) -> (new_chunk, timing) */
static PyObject* py_act(PyObject* self, PyObject* args) {
  PyObject *cap, *im[3], *prop, *out;
  const char* prompt;
  int slot, replan;
  if (!PyArg_ParseTuple(args, "OiOOOOsiO", &cap, &slot, &im[0], &im[1], &im[2], &prop, &prompt, &replan, &out))
    return NULL;
  Pi05Engine* e = get_engine(cap);
  if (!e) return NULL;
  Pi05Image imgs[3];
  Py_buffer iv[3], pv, ov;
  int has[3] = {0, 0, 0};
  PyObject* ret = NULL;
  for (int i = 0; i < 3; ++i)
    if (get_image(im[i], &imgs[i], &iv[i], &has[i]) != 0) goto done_img;
  if (get_f32(prop, &pv, "proprio") != 0) goto done_img;
  if (PyObject_GetBuffer(out, &ov, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) != 0) goto done_prop;
  {
    Pi05Info info;
    pi05_info(e, &info);
    if (ov.len < (Py_ssize_t)(info.action_dim * 4)) {
      PyErr_SetString(PyExc_ValueError, "out buffer too small (float32 [action_dim])");
      goto done_out;
    }
    Pi05Timing t;
    int rc;
    Py_BEGIN_ALLOW_THREADS rc =
        pi05_act(e, slot, imgs, (const float*)pv.buf, (int)(pv.len / 4), prompt, replan, (float*)ov.buf, &t);
    Py_END_ALLOW_THREADS if (rc < 0) {
      PyErr_SetString(PyExc_RuntimeError, pi05_last_error(e));
      goto done_out;
    }
    ret = Py_BuildValue("(iN)", rc, timing_tuple(&t));
  }
done_out:
  PyBuffer_Release(&ov);
done_prop:
  PyBuffer_Release(&pv);
done_img:
  for (int i = 0; i < 3; ++i)
    if (has[i]) PyBuffer_Release(&iv[i]);
  return ret;
}

/* ---- PiBehavior (2025 1st place) ---- */
static PyObject* py_set_task(PyObject* self, PyObject* args) {
  PyObject* cap;
  int slot, task;
  if (!PyArg_ParseTuple(args, "Oii", &cap, &slot, &task)) return NULL;
  Pi05Engine* e = get_engine(cap);
  if (!e) return NULL;
  pi05_set_task(e, slot, task);
  Py_RETURN_NONE;
}

static PyObject* py_set_stage(PyObject* self, PyObject* args) {
  PyObject* cap;
  int slot, stage, mode;
  if (!PyArg_ParseTuple(args, "Oiii", &cap, &slot, &stage, &mode)) return NULL;
  Pi05Engine* e = get_engine(cap);
  if (!e) return NULL;
  pi05_set_stage(e, slot, stage, mode);
  Py_RETURN_NONE;
}

static PyObject* py_get_stage(PyObject* self, PyObject* args) {
  PyObject* cap;
  int slot;
  if (!PyArg_ParseTuple(args, "Oi", &cap, &slot)) return NULL;
  Pi05Engine* e = get_engine(cap);
  if (!e) return NULL;
  int32_t s, p, f;
  pi05_get_stage(e, slot, &s, &p, &f);
  return Py_BuildValue("(iii)", s, p, f);
}

static PyObject* py_pb_config(PyObject* self, PyObject* args) {
  PyObject* cap;
  int ex, keep, steps, tricks;
  if (!PyArg_ParseTuple(args, "Oiiii", &cap, &ex, &keep, &steps, &tricks)) return NULL;
  Pi05Engine* e = get_engine(cap);
  if (!e) return NULL;
  pi05_pb_config(e, ex, keep, steps, tricks);
  Py_RETURN_NONE;
}

static PyObject* py_reset(PyObject* self, PyObject* args) {
  PyObject* cap;
  int slot = -1;
  if (!PyArg_ParseTuple(args, "O|i", &cap, &slot)) return NULL;
  Pi05Engine* e = get_engine(cap);
  if (!e) return NULL;
  pi05_reset(e, slot);
  Py_RETURN_NONE;
}

static PyObject* py_seed(PyObject* self, PyObject* args) {
  PyObject* cap;
  unsigned long long s;
  if (!PyArg_ParseTuple(args, "OK", &cap, &s)) return NULL;
  Pi05Engine* e = get_engine(cap);
  if (!e) return NULL;
  pi05_seed(e, s);
  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"create", py_create, METH_VARARGS, "create(weights_path, device=0) -> engine"},
    {"info", py_info, METH_VARARGS, "info(engine) -> dict"},
    {"infer", py_infer, METH_VARARGS,
     "infer(engine, img0, img1, img2, proprio_f32, prompt, noise_f32|None, out_f64) -> timing"},
    {"act", py_act, METH_VARARGS,
     "act(engine, slot, img0, img1, img2, proprio_f32, prompt, replan_every, out_f32) -> (new_chunk, timing)"},
    {"reset", py_reset, METH_VARARGS, "reset(engine, slot=-1)"},
    {"set_task", py_set_task, METH_VARARGS, "set_task(engine, slot, task)  (PiBehavior)"},
    {"set_stage", py_set_stage, METH_VARARGS,
     "set_stage(engine, slot, stage, mode)  mode 0 = model voting, 1 = fixed from outside (PiBehavior)"},
    {"get_stage", py_get_stage, METH_VARARGS, "get_stage(engine, slot) -> (stage, last predicted, forced)"},
    {"pb_config", py_pb_config, METH_VARARGS, "pb_config(engine, execute, keep, steps, apply_eval_tricks)"},
    {"seed", py_seed, METH_VARARGS, "seed(engine, seed)"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "_pi05native", "pi0.5 native engine (C++/CUDA)", -1, methods};

PyMODINIT_FUNC PyInit__pi05native(void) { return PyModule_Create(&mod); }
