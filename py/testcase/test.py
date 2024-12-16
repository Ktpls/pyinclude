from utilitypack.util_solid import *


@RunThis
def test_CopyNestedClass():
    @dataclasses.dataclass
    class A:
        i: int
        s: str

    @dataclasses.dataclass
    class B:
        a: A
        b: str

    inst = B(
        A(1, "s"),
        "s",
    )
    d = BeanUtil.toMap(inst)
    assert d == {"a": {"i": 1, "s": "s"}, "b": "s"}
    inst: B = BeanUtil.copyProperties(d, B)
    assert inst.a.i == 1


@RunThis
def test_CopyList():
    @dataclasses.dataclass
    class A:
        i: int
        s: str

    @dataclasses.dataclass
    class B:
        la: list[A]
        b: str

    inst = B(
        [A(0, "s"), A(1, "s")],
        "s",
    )
    d = BeanUtil.toMap(inst)
    assert d == {"la": [{"i": 0, "s": "s"}, {"i": 1, "s": "s"}], "b": "s"}
    inst: B = BeanUtil.copyProperties(d, B)
    assert inst.la[1].i == 1
