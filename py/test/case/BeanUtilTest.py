from test.autotest_common import *

# class BeanUtilTest(unittest.TestCase):
# # no furthuer update plan
#     def test_CopySimpleClass(self):
#         @dataclasses.dataclass
#         class A:
#             i: int
#             s: str

#         @dataclasses.dataclass
#         class B:
#             i: int
#             s: str

#         a = A(1, "s")
#         b: B = BeanUtil.copyProperties(a, B)
#         self.assertEqual(b.i, 1)

#     def test_CopyNestedClass(self):
#         @dataclasses.dataclass
#         class A:
#             i: int
#             s: str

#         @dataclasses.dataclass
#         class B:
#             a: A
#             b: str

#         inst = B(
#             A(1, "s"),
#             "s",
#         )
#         m = BeanUtil.toJsonCompatible(inst)
#         self.assertDictEqual(
#             m,
#             {
#                 "a": {
#                     "i": 1,
#                     "s": "s",
#                 },
#                 "b": "s",
#             },
#         )
#         inst: B = BeanUtil.copyProperties(m, B)
#         self.assertEqual(inst.a.i, 1)

#     def test_CopyList(self):
#         @dataclasses.dataclass
#         class A:
#             i: int
#             s: str

#         @dataclasses.dataclass
#         class B:
#             la: list[A]
#             b: str

#         b = B(
#             [A(0, "s"), A(1, "s")],
#             "s",
#         )
#         m = BeanUtil.toJsonCompatible(b)
#         self.assertDictEqual(
#             m,
#             {
#                 "la": [
#                     {"i": 0, "s": "s"},
#                     {"i": 1, "s": "s"},
#                 ],
#                 "b": "s",
#             },
#         )
#         b2: B = BeanUtil.copyProperties(m, B)
#         self.assertEqual(b2.la[1].i, 1)
