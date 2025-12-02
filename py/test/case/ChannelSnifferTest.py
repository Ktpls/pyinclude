from test.autotest_common import *
from unittest.mock import patch


class TestChannelSniffer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sniffer = ChannelSniffer(capacity=10.0, usage=0.0)

    def test_initialization(self):
        """Test that the ChannelSniffer initializes with correct default values."""
        self.assertEqual(self.sniffer.capacity, 10.0)
        self.assertEqual(self.sniffer.usage, 0.0)
        self.assertEqual(self.sniffer.capacity_follow_lambda, 5.0)
        self.assertEqual(self.sniffer.expolite_epsilon, 0.1)

    def test_release_decreases_usage(self):
        """Test that release method decreases usage correctly."""
        self.sniffer.usage = 5.0
        self.sniffer.release(2.0)
        self.assertEqual(self.sniffer.usage, 3.0)

    def test_exploitable_when_under_capacity(self):
        """Test that channel is exploitable when usage + request is under capacity."""
        # Usage 0 + request 5 = 5 which is less than capacity 10
        self.assertTrue(self.sniffer.expolitable(5.0))

        # Usage 8 + request 1 = 9 which is less than capacity 10
        self.sniffer.usage = 8.0
        self.assertTrue(self.sniffer.expolitable(1.0))

    def test_non_exploitable_when_over_capacity_without_epsilon(self):
        """Test that channel is not exploitable when over capacity and epsilon doesn't trigger."""
        self.sniffer.usage = 9.0
        # Usage 9 + request 3 = 12 which is more than capacity 10
        with patch("random.random", return_value=0.5):  # Greater than epsilon 0.1
            self.assertFalse(self.sniffer.expolitable(3.0))

    def test_exploitable_when_over_capacity_with_epsilon(self):
        """Test that channel is exploitable when over capacity but epsilon triggers."""
        self.sniffer.usage = 9.0
        # Usage 9 + request 3 = 12 which is more than capacity 10
        with patch("random.random", return_value=0.05):  # Less than epsilon 0.1
            self.assertTrue(self.sniffer.expolitable(3.0))

    def test_follow_usage_updates_capacity(self):
        """Test that follow_usage updates capacity based on current usage."""
        self.sniffer.capacity = 10.0
        self.sniffer.usage = 12.0  # Over capacity
        old_capacity = self.sniffer.capacity

        self.sniffer.follow_usage()

        # New capacity should be adjusted based on the formula
        expected_capacity = (
            old_capacity
            + (self.sniffer.usage - old_capacity) / self.sniffer.capacity_follow_lambda
        )
        self.assertEqual(self.sniffer.capacity, expected_capacity)

    def test_acquire_success_under_capacity(self):
        """Test successful acquisition when under capacity."""

        class MockSniffer(ChannelSniffer):
            def failable(self, **kw):
                return "success"

        sniffer = MockSniffer(capacity=10.0)
        result = sniffer.acquire(5.0, test_param="value")

        self.assertEqual(result, "success")
        self.assertEqual(sniffer.usage, 5.0)

    def test_acquire_raises_overload_error(self):
        """Test that acquire raises ChannelOverloadError when not exploitable."""

        class MockSniffer(ChannelSniffer):
            def failable(self, **kw):
                return "success"

        sniffer = MockSniffer(capacity=10.0, usage=9.0)

        # With random value greater than epsilon, should raise error
        with patch("random.random", return_value=0.5):
            with self.assertRaises(ChannelSniffer.ChannelOverloadError):
                sniffer.acquire(3.0, test_param="value")

    def test_acquire_follows_usage_when_over_capacity(self):
        """Test that acquire calls follow_usage when usage exceeds capacity."""

        class MockSniffer(ChannelSniffer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.follow_usage_called = False

            def failable(self, **kw):
                return "success"

            def follow_usage(self):
                self.follow_usage_called = True
                super().follow_usage()

        sniffer = MockSniffer(capacity=10.0, usage=9.0)
        # Make it exploitable despite being over capacity
        with patch.object(sniffer, "expolitable", return_value=True):
            sniffer.acquire(2.0, test_param="value")  # Usage becomes 11, over capacity

        self.assertTrue(sniffer.follow_usage_called)
        self.assertEqual(sniffer.usage, 11.0)
