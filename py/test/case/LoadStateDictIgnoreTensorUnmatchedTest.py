from test.autotest_common import *
from utilitypack.util_torch import torch, load_state_dict_ignore_tensor_unmatched


class LoadStateDictIgnoreTensorUnmatchedTest(unittest.TestCase):
    def test_filter_unmatched_tensor(self):
        class ModelA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 10)

        class ModelAVariance(ModelA):
            def __init__(self):
                super().__init__()
                self.linear2 = torch.nn.Linear(10, 3)

        model = ModelA()
        state_dict = model.state_dict()

        model = ModelAVariance()

        load_state_dict_ignore_tensor_unmatched(
            model, state_dict=state_dict, verbose=False
        )
