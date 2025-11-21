from test.autotest_common import *


class StreamTest(unittest.TestCase):

    def test_basic_use(self):
        result = Stream(range(10)).count()
        self.assertEqual(result, 10)

    def test_map(self):
        result = Stream([1, 2, 3]).map(lambda x: x * 2).to_list()
        self.assertEqual(result, [2, 4, 6])

    def test_flat_map(self):
        result = Stream([[1, 2], [3, 4]]).flat_map(lambda x: Stream(x)).to_list()
        self.assertEqual(result, [1, 2, 3, 4])

    def test_filter(self):
        result = Stream([1, 2, 3, 4]).filter(lambda x: x % 2 == 0).to_list()
        self.assertEqual(result, [2, 4])

    def test_for_each(self):
        result = 0

        def add_to_result(x):
            nonlocal result
            result += x

        Stream(range(4)).for_each(add_to_result)
        self.assertEqual(result, 6)

    def test_peek(self):
        seen = []
        result = Stream([1, 2, 3]).peek(lambda x: seen.append(x)).to_list()
        self.assertEqual(result, [1, 2, 3])
        self.assertEqual(seen, [1, 2, 3])

    def test_distinct(self):
        result = Stream([1, 2, 2, 3]).distinct().to_list()
        self.assertEqual(result, [1, 2, 3])

    def test_sorted(self):
        result = Stream([3, 1, 2]).sorted().to_list()
        self.assertEqual(result, [1, 2, 3])

    def test_count(self):
        result = Stream([1, 2, 3]).count()
        self.assertEqual(result, 3)

    def test_sum(self):
        result = Stream([1, 2, 3]).sum()
        self.assertEqual(result, 6)

    def test_group_by(self):
        result = Stream([1, 2, 3, 4]).group_by(lambda x: x % 2)
        self.assertEqual(result[0], [2, 4])
        self.assertEqual(result[1], [1, 3])

    def test_reduce(self):
        result = Stream([1, 2, 3, 4]).reduce(lambda a, b: a + b)
        self.assertEqual(result, 10)

    def test_limit(self):
        result = Stream(range(100)).limit(5).to_list()
        self.assertEqual(result, [0, 1, 2, 3, 4])

    def test_skip(self):
        result = Stream([0, 1, 2, 3, 4]).skip(2).to_list()
        self.assertEqual(result, [2, 3, 4])

    def test_min_max(self):
        s = lambda: Stream([3, 1, 4, 1, 5])
        self.assertEqual(s().min(), 1)
        self.assertEqual(s().max(), 5)
        self.assertTupleEqual(s().minmax(), (1, 5))

    def test_find_first(self):
        self.assertEqual(Stream([1, 2, 3]).find_first(), 1)
        self.assertIsNone(Stream([]).find_first())

    def test_any_all_none_match(self):
        s = lambda: Stream([2, 4, 6])
        self.assertTrue(s().any_match(lambda x: x % 2 == 0))
        self.assertTrue(s().all_match(lambda x: x % 2 == 0))
        self.assertTrue(s().none_match(lambda x: x > 10))

        s2 = lambda: Stream([1, 3, 5])
        self.assertFalse(s2().any_match(lambda x: x % 2 == 0))
        self.assertFalse(s2().all_match(lambda x: x % 2 == 0))
        self.assertFalse(s2().none_match(lambda x: x <= 1))

    def test_to_dict(self):
        result = Stream([(1, "a"), (2, "b")]).to_dict(lambda x: x[0], lambda x: x[1])
        self.assertEqual(result, {1: "a", 2: "b"})

    def test_to_set(self):
        result = Stream([1, 1, 2]).to_set()
        self.assertSetEqual(result, {1, 2})

    def test_gather_async(self):
        import asyncio

        async def async_task(x: int):
            await asyncio.sleep(0)
            return x + 1

        result = Stream([1, 2, 3]).map(async_task).gather_async().to_list()
        self.assertEqual(result, [2, 3, 4])

    def test_gather_thread(self):
        import concurrent.futures

        def task(x: int):
            return x + 1

        pool = concurrent.futures.ThreadPoolExecutor()
        result = (
            Stream([1, 2, 3])
            .map(lambda x: pool.submit(task, x=x))
            .gather_thread_future()
            .collect(list)
        )
        self.assertEqual(result, [2, 3, 4])

    def test_reversed(self):
        result = Stream([1, 2, 3]).reversed().to_list()
        self.assertEqual(result, [3, 2, 1])

    def test_auto_unpacked(self):
        result = Stream(range(4)).wrap_iterator(enumerate).map(lambda a, b: b).sum()
        self.assertEqual(result, 6)

    def test_auto_awaited(self):
        async def add1(x: int):
            return x + 1

        result = Stream(range(3)).map(add1).map(add1).gather_async().sum()
        self.assertEqual(result, 9)

    def test_auto_unpacked_and_awaited(self):
        async def add1(x: int, y: int):
            return x + 1, y + 1

        async def sum(x: int, y: int):
            return x + y

        result = (
            Stream(range(3))
            .wrap_iterator(enumerate)
            .map(add1)
            .map(sum)
            .gather_async()
            .sum()
        )
        self.assertEqual(result, 12)

    def test_collect_to_string_io(self):
        result = Stream(range(3)).map(str).collect(Stream.Collectors.stringIo())
        result.seek(0)
        result = result.read()
        self.assertEqual(result, "012")

    def test_collect_to_print(self):
        buf = io.StringIO()
        Stream(range(3)).collect(Stream.Collectors.print(end="", flush=True, file=buf))
        buf.seek(0)
        result = buf.read()
        self.assertEqual(result, "012")
