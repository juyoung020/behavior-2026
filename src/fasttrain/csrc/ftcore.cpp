// torch 확장 바인딩 (파이썬 ↔ C++ 엔진). GIL 을 풀고 엔진을 돌려서, 파이썬 쪽 학습 루프와 겹쳐 돌 수 있게 한다.
#include <torch/extension.h>

#include "engine.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "NVDEC + 색 변환 표 + JAX 순서 크기 조정 (π0.5 학습 데이터 가속)";
    py::class_<ft::Engine>(m, "Engine", py::dynamic_attr())
        .def(py::init([](const std::vector<std::string>& videos, const std::vector<std::string>& indexes,
                         torch::Tensor lut, const std::vector<int>& sizes, torch::Tensor start, torch::Tensor count,
                         torch::Tensor weight, torch::Tensor split_rows, torch::Tensor split_cols, int threads,
                         int device) {
                 TORCH_CHECK(lut.is_cuda() && lut.dtype() == torch::kUInt8 && lut.numel() == (1LL << 24) * 3,
                             "lut 는 GPU uint8 [2^24*3]");
                 TORCH_CHECK(lut.is_contiguous());
                 auto s = start.to(torch::kInt32).contiguous().cpu();
                 auto c = count.to(torch::kInt32).contiguous().cpu();
                 auto w = weight.to(torch::kFloat32).contiguous().cpu();
                 auto sr = split_rows.to(torch::kInt32).contiguous().cpu();
                 auto sc = split_cols.to(torch::kInt32).contiguous().cpu();
                 const int64_t k = (int64_t)sizes.size() * 224;
                 TORCH_CHECK(s.numel() == k && c.numel() == k && w.numel() == k * 8 && sr.numel() == k &&
                             sc.numel() == k, "탭 표 크기");
                 return new ft::Engine(videos, indexes, (uintptr_t)lut.data_ptr(), sizes, s.data_ptr<int32_t>(),
                                       c.data_ptr<int32_t>(), w.data_ptr<float>(), sr.data_ptr<int32_t>(),
                                       sc.data_ptr<int32_t>(), threads, device);
             }),
             py::arg("videos"), py::arg("indexes"), py::arg("lut"), py::arg("sizes"), py::arg("start"),
             py::arg("count"), py::arg("weight"), py::arg("split_rows"), py::arg("split_cols"),
             py::arg("threads") = 8, py::arg("device") = 0)
        .def(
            "run",
            [](ft::Engine& e, torch::Tensor req, torch::Tensor out, int mode) {
                TORCH_CHECK(req.dtype() == torch::kInt64 && req.dim() == 2 && req.size(1) == 2, "req: int64 [n,2]");
                TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kUInt8 && out.is_contiguous(), "out: GPU uint8");
                auto r = req.contiguous().cpu();
                int n = (int)r.size(0);
                if (mode == 0) TORCH_CHECK(out.numel() == (int64_t)n * 224 * 224 * 3, "out 크기");
                py::gil_scoped_release nogil;
                e.run(r.data_ptr<int64_t>(), n, (uintptr_t)out.data_ptr(), mode);
            },
            py::arg("req"), py::arg("out"), py::arg("mode") = 0)
        .def(
            "resize",
            [](ft::Engine& e, torch::Tensor in, torch::Tensor out) {
                TORCH_CHECK(in.is_cuda() && in.dtype() == torch::kUInt8 && in.is_contiguous() && in.dim() == 4);
                TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kUInt8 && out.is_contiguous());
                TORCH_CHECK(out.numel() == in.size(0) * 224 * 224 * 3, "out 크기");
                py::gil_scoped_release nogil;
                e.resize((uintptr_t)in.data_ptr(), (int)in.size(0), (int)in.size(2), (uintptr_t)out.data_ptr());
            },
            py::arg("rgb"), py::arg("out"))
        .def("info", &ft::Engine::info)
        .def("pts", &ft::Engine::pts)
        .def("stats", &ft::Engine::stats);
}
