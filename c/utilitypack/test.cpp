#include "util_solid.h"

void test_pipe() {
    pipe<int>::range(10)
        .sorted([](auto a, auto b) { return a > b; })
        .groupby<int>([](auto i) { return i % 3; })
        .foreach(
            [](auto i) {
                std::cout << i.first << std::endl;
                pipe<int>::of(i.second).foreach([](auto j) { std::cout << "\t" << j << std::endl; });
            })
        .flatmap<int>([](auto i) { return pipe<int>(i.second); })
        .foreach([](auto i) { std::cout << i << std::endl; });

    pipe<int>::range(10)
        .map<int>([](auto i) { return i * i; })
        .enumerate()
        .reduce<std::map<int, int>>(
            [](std::map<int, int>& a, std::pair<int, int> b) -> std::map<int, int> & {
                a[b.first] = b.second;
                return a;
            },
            std::map<int, int>());
}

void test_singlesectionedtimer() {
    SingleSectionedTimer timer(true); // 创建一个计时器实例并立即启动

    // 模拟一些操作
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 输出已经过去的时间
    std::cout << "Elapsed time: " << timer.get() << " seconds" << std::endl;

    // 重启计时器
    timer.clear();
    timer.start();

    // 再次模拟一些操作
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 再次输出已经过去的时间
    std::cout << "Elapsed time after restart: " << timer.get() << " seconds" << std::endl;

}

int main() {
    test_singlesectionedtimer();
    return 0;
}