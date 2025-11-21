from test.autotest_common import *


class ExpNodeReorganizationTest(unittest.TestCase):
    def assert_reorg_opr_result(
        self, l: list[expparser.Ast.Element], expected: expparser.Ast.Element
    ):
        r = expparser._FlatOperatorOrganizer.reorganize_operator_sort(
            expparser._FlatOperatorOrganizer.Ir.lof(l)
        ).elm
        self.assertDictEqual(dataclasses.asdict(r), dataclasses.asdict(expected))

    def print_reorg_opr_result(self, l: list[expparser.Ast.Element]):
        r = expparser._FlatOperatorOrganizer.reorganize_operator_sort(
            expparser._FlatOperatorOrganizer.Ir.lof(l)
        ).elm
        pprint.pp(r)

    def test_1p2(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Object(2, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Object(val=1, sec=None),
                    expparser.Ast.Object(val=2, sec=None),
                ],
            ),
        )

    def test_1p2p3(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Object(2, None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Object(3, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.ADD,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=1, sec=None),
                            expparser.Ast.Object(val=2, sec=None),
                        ],
                    ),
                    expparser.Ast.Object(val=3, sec=None),
                ],
            ),
        )

    def test_1p2m3(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Object(2, None),
                expparser.Ast.Operator(expparser._OprType.MUL, None, None),
                expparser.Ast.Object(3, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Object(val=1, sec=None),
                    expparser.Ast.Operator(
                        val=expparser._OprType.MUL,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=2, sec=None),
                            expparser.Ast.Object(val=3, sec=None),
                        ],
                    ),
                ],
            ),
        )

    def test_1m2p3(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.Operator(expparser._OprType.MUL, None, None),
                expparser.Ast.Object(2, None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Object(3, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.MUL,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=1, sec=None),
                            expparser.Ast.Object(val=2, sec=None),
                        ],
                    ),
                    expparser.Ast.Object(val=3, sec=None),
                ],
            ),
        )

    def test_n1(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.NEG,
                sec=None,
                operands=[expparser.Ast.Object(val=1, sec=None)],
            ),
        )

    def test_nn1(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.NEG,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.NEG,
                        sec=None,
                        operands=[expparser.Ast.Object(val=1, sec=None)],
                    )
                ],
            ),
        )

    def test_n1pow2(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
                expparser.Ast.Operator(expparser._OprType.POW, None, None),
                expparser.Ast.Object(2, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.NEG,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.POW,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=1, sec=None),
                            expparser.Ast.Object(val=2, sec=None),
                        ],
                    )
                ],
            ),
        )

    def test_n1pn1(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.NEG,
                        sec=None,
                        operands=[expparser.Ast.Object(val=1, sec=None)],
                    ),
                    expparser.Ast.Operator(
                        val=expparser._OprType.NEG,
                        sec=None,
                        operands=[expparser.Ast.Object(val=1, sec=None)],
                    ),
                ],
            ),
        )

    def test_call(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.ArgumentTuple([], None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.CALL,
                sec=None,
                operands=[
                    expparser.Ast.Object(val=1, sec=None),
                    expparser.Ast.ArgumentTuple(val=[], sec=None),
                ],
            ),
        )

    def test_neg_call(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
                expparser.Ast.ArgumentTuple([], None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.NEG,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.CALL,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=1, sec=None),
                            expparser.Ast.ArgumentTuple(val=[], sec=None),
                        ],
                    )
                ],
            ),
        )

    def test_callp1(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.ArgumentTuple([], None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Object(1, None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.CALL,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=1, sec=None),
                            expparser.Ast.ArgumentTuple(val=[], sec=None),
                        ],
                    ),
                    expparser.Ast.Object(1, None),
                ],
            ),
        )

    def test_call_call(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Object(1, None),
                expparser.Ast.ArgumentTuple([], None),
                expparser.Ast.ArgumentTuple([], None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.CALL,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.CALL,
                        sec=None,
                        operands=[
                            expparser.Ast.Object(val=1, sec=None),
                            expparser.Ast.ArgumentTuple(val=[], sec=None),
                        ],
                    ),
                    expparser.Ast.ArgumentTuple(val=[], sec=None),
                ],
            ),
        )

    def test_complex(self):
        self.assert_reorg_opr_result(
            [
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
                expparser.Ast.ArgumentTuple([], None),
                expparser.Ast.Operator(expparser._OprType.ADD, None, None),
                expparser.Ast.Operator(expparser._OprType.NEG, None, None),
                expparser.Ast.Object(1, None),
                expparser.Ast.ArgumentTuple([], None),
            ],
            expparser.Ast.Operator(
                val=expparser._OprType.ADD,
                sec=None,
                operands=[
                    expparser.Ast.Operator(
                        val=expparser._OprType.NEG,
                        sec=None,
                        operands=[
                            expparser.Ast.Operator(
                                val=expparser._OprType.CALL,
                                sec=None,
                                operands=[
                                    expparser.Ast.Object(val=1, sec=None),
                                    expparser.Ast.ArgumentTuple(val=[], sec=None),
                                ],
                            )
                        ],
                    ),
                    expparser.Ast.Operator(
                        val=expparser._OprType.NEG,
                        sec=None,
                        operands=[
                            expparser.Ast.Operator(
                                val=expparser._OprType.CALL,
                                sec=None,
                                operands=[
                                    expparser.Ast.Object(val=1, sec=None),
                                    expparser.Ast.ArgumentTuple(val=[], sec=None),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        )
