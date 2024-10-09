#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <map>
#include <functional>
template <typename eT>
class pipe {
private:
public:
    std::vector<eT> c;
    pipe(const std::vector<eT>& c_) : c(c_) {}
    pipe(std::vector<eT>&& c_) : c(std::move(c_)) {}
    pipe(const pipe& c_) : c(c_.c) {}
    pipe(pipe&& c_) = default;
    static auto of(const std::vector<eT>& c_) { return pipe(c_); }
    static auto of(const std::initializer_list<eT>& c_) { return pipe<eT>(std::vector<eT>(c_)); }
    template <typename kT, typename vT>
    static auto of(const std::map<kT, vT>& c_) {
        auto c = std::vector<std::pair<kT, vT>>();
        c.reserve(c_.size());
        for (auto p : c_)
            c.push_back(p);
        return pipe<std::pair<kT, vT>>(c);
    }
    size_t size() { return c.size(); }
    eT& operator[](size_t i) { return c[i]; }
    static auto range(eT beg, eT end, eT step) {
        auto ret = std::vector<eT>();
        for (eT x = beg; step > 0 ? x < end : x > end; x += step)
            ret.push_back(x);
        return pipe(ret);
    }
    static auto range(eT beg, eT end) { return pipe::range(beg, end, 1); }
    static auto range(eT end) { return pipe::range(0, end); }
    pipe& foreach(std::function<void(eT&)> consumer) {
        for (auto& i : c)
            consumer(i);
        return *this;
    }
    pipe& filter(std::function<bool(const eT&)> predicate) {
        auto it = std::copy_if(c.begin(), c.end(), c.begin(), predicate);
        c.resize(std::distance(c.begin(), it));
        return *this;
    }
    template <typename rT>
    auto map(std::function<rT(const eT&)> mapper) {
        auto ret = std::vector<rT>();
        ret.reserve(c.size());
        this->foreach([&ret, &mapper](auto i) { ret.push_back(mapper(i)); });
        return pipe<rT>(ret);
    }
    auto enumerate() {
        auto ret = std::vector<std::pair<size_t, eT>>();
        ret.reserve(c.size());
        for (size_t i = 0; i < c.size(); i++)
            ret.push_back(std::make_pair(i, c[i]));
        return pipe<std::pair<size_t, eT>>(ret);
    }
    template <typename kT>
    auto groupby(std::function<kT(const eT&)> keySelector) {
        auto ret = std::map<kT, std::vector<eT>>();
        for (auto& i : c) {
            auto k = keySelector(i);
            auto pos = ret.find(k);
            if (pos == ret.end())
                pos = ret.insert(std::make_pair(k, std::vector<eT>())).first;
            pos->second.push_back(i);
        }
        return pipe::of<kT, std::vector<eT>>(ret);
    }
    template <typename rT>
    auto flatmap(std::function<pipe<rT>(const eT&)> mapper) {
        auto ret = std::vector<rT>();
        this->map<pipe<rT>>(mapper)
            .foreach([&ret](auto i) { i.foreach([&ret](auto j) { ret.push_back(j); }); });
        return pipe<rT>(ret);
    }
    auto sorted(std::function<bool(const eT&, const eT&)> opr) {
        std::sort(this->c.begin(), this->c.end(), opr);
        return *this;
    }
    template <typename rT>
    auto reduce(std::function<rT& (rT&, eT)> pred, rT&& init) {
        rT& reduced = init;
        this->foreach([&reduced, &pred](auto e) { reduced = pred(reduced, e); });
        return reduced;
    }
    ~pipe() {}
};