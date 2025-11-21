from test.autotest_common import *


class ReadWriteLockTest(unittest.TestCase):
    def setUp(self):
        self.rwlock = ReadWriteLock()
        self.shared_resource = 0
        self.read_count = 0
        self.write_count = 0

    def test_init(self):
        """测试初始化"""
        self.assertIsInstance(self.rwlock, ReadWriteLock)

    def test_acquire_release_read_lock(self):
        """测试获取和释放读锁"""
        self.rwlock.acquire_read()
        # 验证内部状态
        self.assertEqual(self.rwlock._readers, 1)
        self.rwlock.release_read()
        self.assertEqual(self.rwlock._readers, 0)

    def test_acquire_release_write_lock(self):
        """测试获取和释放写锁"""
        self.rwlock.acquire_write()
        self.assertEqual(self.rwlock._writers, 1)
        self.assertEqual(self.rwlock._pending_writers, 0)
        self.rwlock.release_write()
        self.assertEqual(self.rwlock._writers, 0)

    def test_multiple_readers(self):
        """测试多个读锁可以同时获取"""

        def reader():
            self.rwlock.acquire_read()
            time.sleep(0.1)  # 模拟读操作
            self.rwlock.release_read()

        threads = []
        for _ in range(5):
            t = threading.Thread(target=reader)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 验证所有读锁都已释放
        self.assertEqual(self.rwlock._readers, 0)

    def test_writer_blocks_reader(self):
        """测试写锁会阻塞读锁"""
        # 获取写锁
        self.rwlock.acquire_write()

        reader_acquired = threading.Event()

        def reader():
            self.rwlock.acquire_read()
            reader_acquired.set()
            self.rwlock.release_read()

        # 启动读线程
        t = threading.Thread(target=reader)
        t.start()

        # 等待一小段时间，确保读线程尝试获取锁
        time.sleep(0.1)

        # 读线程应该被阻塞
        self.assertFalse(reader_acquired.is_set())

        # 释放写锁
        self.rwlock.release_write()

        # 等待读线程完成
        t.join(timeout=1)

        # 读线程现在应该能获取到锁
        self.assertTrue(reader_acquired.is_set())

    def test_reader_blocks_writer(self):
        """测试读锁会阻塞写锁"""
        # 获取读锁
        self.rwlock.acquire_read()

        writer_acquired = threading.Event()

        def writer():
            self.rwlock.acquire_write()
            writer_acquired.set()
            self.rwlock.release_write()

        # 启动写线程
        t = threading.Thread(target=writer)
        t.start()

        # 等待一小段时间，确保写线程尝试获取锁
        time.sleep(0.1)

        # 写线程应该被阻塞
        self.assertFalse(writer_acquired.is_set())

        # 释放读锁
        self.rwlock.release_read()

        # 等待写线程完成
        t.join(timeout=1)

        # 写线程现在应该能获取到锁
        self.assertTrue(writer_acquired.is_set())

    def test_context_manager_read(self):
        """测试读锁的上下文管理器"""
        with self.rwlock.gen_rlock():
            self.assertEqual(self.rwlock._readers, 1)
        self.assertEqual(self.rwlock._readers, 0)

    def test_context_manager_write(self):
        """测试写锁的上下文管理器"""
        with self.rwlock.gen_wlock():
            self.assertEqual(self.rwlock._writers, 1)
        self.assertEqual(self.rwlock._writers, 0)

    def test_write_preference(self):
        """测试写锁优先（避免写锁饥饿）"""
        # 先让一些读锁等待
        reader_events = [threading.Event() for _ in range(3)]
        writer_event = threading.Event()

        def reader(i):
            with self.rwlock.gen_rlock():
                reader_events[i].set()

        def writer():
            with self.rwlock.gen_wlock():
                writer_event.set()

        # 先启动写线程（让它排队）
        writer_thread = threading.Thread(target=writer)
        writer_thread.start()

        # 等待一小段时间确保写线程已经开始等待
        time.sleep(0.1)

        # 然后启动读线程
        reader_threads = []
        for i in range(3):
            t = threading.Thread(target=reader, args=(i,))
            reader_threads.append(t)
            t.start()

        # 给一点时间让所有读线程尝试获取锁
        time.sleep(0.1)

        # 写线程应该在读线程之前获得锁
        self.assertTrue(writer_event.is_set())

        # 等待所有线程完成
        writer_thread.join()
        for t in reader_threads:
            t.join()
