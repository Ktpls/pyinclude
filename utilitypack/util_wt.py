from sympy import false
from .util_windows import *
import requests

import json


def GetWtHwnd():
    ret = win32gui.FindWindow("DagorWClass", None)
    if ret == win32con.NULL:
        raise Exception("FindWindow() failed")
    return ret


@Singleton
class WarthunderWindow(Cache):
    def __init__(self):
        def toFetch():
            try:
                return GetWtHwnd()
            except:
                return None

        super().__init__(
            toFetch=toFetch,
            updateStrategey=Cache.UpdateStrategey.Outdated(10),
        )

    def isValid(self):
        hwnd = self.get()
        try:
            return hwnd is not None and hwnd != 0 and win32gui.IsWindow(hwnd) != 0
        except:
            return False

    def isFocus(self):
        try:
            fore = win32gui.GetForegroundWindow()
            return self.isValid() and fore == self.get()
        except Exception as e:
            return False


class Port8111:
    class FetchFailure(Exception): ...

    class ValidBean:
        def expectValid(self):
            if not self.valid:
                raise Port8111.FetchFailure()
            return self

    class BeanInvalid(ValidBean):
        def expectValid(self):
            raise Port8111.FetchFailure()

    class BeanIndicatorBase:
        class IndicatorType(enum.Enum):
            air = 0
            tank = 1

        def expectToBe(self, type):
            if (
                type == Port8111.BeanIndicatorBase.IndicatorType.air
                and not isinstance(self, Port8111.BeanIndicatorAir)
            ) or (
                type == Port8111.BeanIndicatorBase.IndicatorType.tank
                and not isinstance(self, Port8111.BeanIndicatorTank)
            ):
                raise Port8111.FetchFailure()
            return self

    # consider get an easy way to collect all possible fields for various vehicles
    @dataclasses.dataclass
    class BeanIndicatorAir(BeanIndicatorBase, ValidBean):
        valid: bool = None
        army: str = None
        type: str = None
        speed: float = None
        ammo_counter1: float = None
        ammo_counter1_lamp: float = None
        ammo_counter2: float = None
        ammo_counter2_lamp: float = None
        ammo_counter3: float = None
        ammo_counter3_lamp: float = None
        ammo_counter4: float = None
        ammo_counter5: float = None
        ammo_counter6: float = None
        ammo_counter7: float = None
        ammo_counter8: float = None
        oxygen: float = None
        prop_pitch_hour: float = None
        prop_pitch_min: float = None
        pedals1: float = None
        pedals2: float = None
        pedals3: float = None
        pedals4: float = None
        pedals5: float = None
        pedals6: float = None
        pedals7: float = None
        pedals8: float = None
        stick_elevator: float = None
        stick_elevator1: float = None
        stick_ailerons: float = None
        vario: float = None
        altitude_10k: float = None
        altitude_hour: float = None
        altitude_min: float = None
        aviahorizon_roll: float = None
        aviahorizon_pitch: float = None
        bank: float = None
        bank1: float = None
        bank2: float = None
        turn: float = None
        compass: float = None
        compass1: float = None
        clock_hour: float = None
        clock_min: float = None
        clock_sec: float = None
        manifold_pressure: float = None
        manifold_pressure1: float = None
        rpm: float = None
        rpm_min: float = None
        rpm1_min: float = None
        rpm_hour: float = None
        rpm1_hour: float = None
        oil_pressure: float = None
        oil_pressure1: float = None
        oil_temperature: float = None
        oil_temperature1: float = None
        oil_temperature2: float = None
        oil_temperature3: float = None
        mixture: float = None
        mixture1: float = None
        mixture_1: float = None
        carb_temperature: float = None
        carb_temperature1: float = None
        fuel: float = None
        fuel1: float = None
        fuel2: float = None
        fuel_pressure: float = None
        fuel_pressure1: float = None
        gears: float = None
        gears1: float = None
        gear_lamp_down: float = None
        gear_lamp_off: float = None
        gear_lamp_up: float = None
        flaps: float = None
        trimmer: float = None
        throttle: float = None
        throttle_1: float = None
        weapon1: float = None
        weapon2: float = None
        weapon3: float = None
        prop_pitch: float = None
        prop_pitch1: float = None
        supercharger: float = None
        radiator: float = None
        oil_radiator_indicator: float = None
        oil_radiator_lever: float = None
        oil_radiator_lever1_1: float = None
        radiator_indicator: float = None
        radiator_lever1_1: float = None
        water_temperature: float = None
        blister1: float = None
        blister2: float = None
        blister3: float = None

    @dataclasses.dataclass
    class BeanIndicatorTank(BeanIndicatorBase, ValidBean):
        valid: bool = None
        army: str = None
        type: str = None
        stabilizer: float = None
        gear: float = None
        gear_neutral: float = None
        speed: float = None
        has_speed_warning: float = None
        rpm: float = None
        driving_direction_mode: float = None
        cruise_control: float = None
        lws: float = None
        ircm: float = None
        roll_indicators_is_available: float = None
        first_stage_ammo: float = None
        crew_total: float = None
        crew_current: float = None
        crew_distance: float = None
        gunner_state: float = None
        driver_state: float = None

    @dataclasses.dataclass
    class BeanMapInfo:
        grid_size: list[float] = None
        grid_steps: list[float] = None
        grid_zero: list[float] = None
        hud_type: int = None
        map_generation: int = None
        map_max: list[float] = None
        map_min: list[float] = None
        valid: bool = None

    @dataclasses.dataclass
    class BeanState(ValidBean):
        @dataclasses.dataclass
        class UnitValue:
            unit: str
            value: list[float]

        @dataclasses.dataclass
        class Engine:
            throttle: "Port8111.BeanState.UnitValue" = None
            RPM_throttle: "Port8111.BeanState.UnitValue" = None
            mixture: "Port8111.BeanState.UnitValue" = None
            radiator: "Port8111.BeanState.UnitValue" = None
            compressor_stage: "Port8111.BeanState.UnitValue" = None
            magneto: "Port8111.BeanState.UnitValue" = None
            power: "Port8111.BeanState.UnitValue" = None
            RPM: "Port8111.BeanState.UnitValue" = None
            manifold_pressure: "Port8111.BeanState.UnitValue" = None
            oil_temp: "Port8111.BeanState.UnitValue" = None
            pitch: "Port8111.BeanState.UnitValue" = None
            thrust: "Port8111.BeanState.UnitValue" = None
            efficiency: "Port8111.BeanState.UnitValue" = None
            water_temp: "Port8111.BeanState.UnitValue" = None

        valid: bool = None
        aileron: UnitValue = None
        elevator: UnitValue = None
        rudder: UnitValue = None
        flaps: UnitValue = None
        gear: UnitValue = None
        H: UnitValue = None
        TAS: UnitValue = None
        IAS: UnitValue = None
        M: UnitValue = None
        AoA: UnitValue = None
        AoS: UnitValue = None
        Ny: UnitValue = None
        Vy: UnitValue = None
        Wx: UnitValue = None
        Mfuel: UnitValue = None
        Mfuel0: UnitValue = None
        engine: Engine = dataclasses.field(default_factory=Engine)

        @staticmethod
        def fromDict(data_dict: typing.Dict[str, float]) -> "Port8111.BeanState":
            @dataclasses.dataclass
            class keyFormat:
                name: str
                index: int
                unit: str
                originalKey: str

                @staticmethod
                def parse_key_format(key: str):
                    if ", " in key:
                        nameindex, unit = key.split(", ")
                    else:
                        nameindex = key
                        unit = None
                    if " " in nameindex:
                        namesindex = nameindex.split(" ")
                        if namesindex[-1].isdigit():
                            name = nameindex[
                                : -(len(namesindex[-1]) + 1)
                            ]  # +1 for space
                            index = int(namesindex[-1])
                        else:
                            name = " ".join(namesindex)
                            index = None
                    else:
                        name = nameindex
                        index = None
                    return keyFormat(name, index, unit, key)

            valid = data_dict.get("valid", False)
            aircraft_fields = {"valid": valid}
            if valid:
                keys = [
                    keyFormat.parse_key_format(k)
                    for k in data_dict.keys()
                    if k != "valid"
                ]
                keys.sort(key=lambda x: x.name)
                grouped_keys = {
                    key: list(_) for key, _ in itertools.groupby(keys, lambda k: k.name)
                }

                # Initialize corresponding fields in Aircraft
                for name, keys in grouped_keys.items():
                    # Assuming all units are the same for each key in the same group
                    unit = keys[0].unit
                    values = [data_dict[key.originalKey] for key in keys]
                    aircraft_fields[name] = Port8111.BeanState.UnitValue(unit, values)

                # Create Engine instance
                engine = Port8111.BeanState.Engine(
                    throttle=aircraft_fields.get("throttle", None),
                    RPM_throttle=aircraft_fields.get("RPM throttle", None),
                    mixture=aircraft_fields.get("mixture", None),
                    radiator=aircraft_fields.get("radiator", None),
                    compressor_stage=aircraft_fields.get("compressor stage", None),
                    magneto=aircraft_fields.get("magneto", None),
                    power=aircraft_fields.get("power", None),
                    RPM=aircraft_fields.get("RPM", None),
                    manifold_pressure=aircraft_fields.get("manifold pressure", None),
                    oil_temp=aircraft_fields.get("oil temp", None),
                    pitch=aircraft_fields.get("pitch", None),
                    thrust=aircraft_fields.get("thrust", None),
                    efficiency=aircraft_fields.get("efficiency", None),
                    water_temp=aircraft_fields.get("water temp", None),
                )

                def getFromAircraftFields_NoniterableValueDesired(name):
                    ret = aircraft_fields.get(name, None)
                    if (
                        ret is not None
                        and isinstance(ret, Port8111.BeanState.UnitValue)
                        and isinstance(ret.value, typing.Iterable)
                    ):
                        ret.value = ret.value[0]
                    return ret

                # Create Aircraft instance
                aircraft = Port8111.BeanState(
                    valid=getFromAircraftFields_NoniterableValueDesired("valid"),
                    aileron=getFromAircraftFields_NoniterableValueDesired("aileron"),
                    elevator=getFromAircraftFields_NoniterableValueDesired("elevator"),
                    rudder=getFromAircraftFields_NoniterableValueDesired("rudder"),
                    flaps=getFromAircraftFields_NoniterableValueDesired("flaps"),
                    gear=getFromAircraftFields_NoniterableValueDesired("gear"),
                    H=getFromAircraftFields_NoniterableValueDesired("H"),
                    TAS=getFromAircraftFields_NoniterableValueDesired("TAS"),
                    IAS=getFromAircraftFields_NoniterableValueDesired("IAS"),
                    M=getFromAircraftFields_NoniterableValueDesired("M"),
                    AoA=getFromAircraftFields_NoniterableValueDesired("AoA"),
                    AoS=getFromAircraftFields_NoniterableValueDesired("AoS"),
                    Ny=getFromAircraftFields_NoniterableValueDesired("Ny"),
                    Vy=getFromAircraftFields_NoniterableValueDesired("Vy"),
                    Wx=getFromAircraftFields_NoniterableValueDesired("Wx"),
                    Mfuel=getFromAircraftFields_NoniterableValueDesired("Mfuel"),
                    Mfuel0=getFromAircraftFields_NoniterableValueDesired("Mfuel0"),
                    engine=engine,
                )
            else:
                aircraft = Port8111.BeanState(
                    valid=valid,
                )
            return aircraft

    @dataclasses.dataclass
    class BeanHudMsgDamage:
        id: int
        msg: str
        sender: str
        enemy: bool
        mode: str
        time: int  # in seconds

    @dataclasses.dataclass
    class BeanHudMsg:
        events: list
        damage: list["Port8111.BeanHudMsgDamage"]

        @staticmethod
        def fromDict(data_dict: dict):
            # return BeanUtil.copyProperties(data_dict, Port8111.BeanHudMsg)
            events = data_dict.get("events", list())
            damage = data_dict.get("damage", list())
            damage = [
                BeanUtil.copyProperties(d, Port8111.BeanHudMsgDamage) for d in damage
            ]
            return Port8111.BeanHudMsg(events=events, damage=damage)

    class QueryType(enum.Enum):
        indicator = 0
        map_info = 1
        state = 2
        hudmsg = 3

        @staticmethod
        def __throwEnumNotFound():
            raise Exception("enum not found")

        def getPath(self):
            if self == self.indicator:
                return "indicators"
            elif self == self.map_info:
                return "map_info.json"
            elif self == self.state:
                return "state"
            elif self == self.hudmsg:
                return "hudmsg"
            Port8111.QueryType.__throwEnumNotFound()
            return ""

        def parseJson(self, json_obj):
            if json_obj is None:
                # may happen when reading json failed
                return Port8111.BeanInvalid()
            if self == self.indicator:
                if json_obj["valid"] == False:
                    return BeanUtil.copyProperties(json_obj, Port8111.BeanIndicatorAir)
                army = json_obj["army"]
                if army == "air":
                    return BeanUtil.copyProperties(json_obj, Port8111.BeanIndicatorAir)
                elif army == "tank":
                    return BeanUtil.copyProperties(json_obj, Port8111.BeanIndicatorTank)
                Port8111.QueryType.__throwEnumNotFound()
            elif self == self.map_info:
                return BeanUtil.copyProperties(json_obj, Port8111.BeanMapInfo)
            elif self == self.state:
                return Port8111.BeanState.fromDict(json_obj)
            elif self == self.hudmsg:
                return Port8111.BeanHudMsg.fromDict(json_obj)
            Port8111.QueryType.__throwEnumNotFound()

    @staticmethod
    def get_raw_json(queryType: "Port8111.QueryType", param=None, timeout=None):
        try:
            response = requests.get(
                "http://127.0.0.1:8111/" + queryType.getPath(),
                params=param,
                timeout=timeout if timeout is not None else 0.5,
            )
        except (requests.ConnectionError, requests.ReadTimeout):
            return None
        json_data = response.json()
        return json_data

    @staticmethod
    def get(
        queryType: "Port8111.QueryType",
        param=None,
        timeout: typing.Optional[float] = None,
    ) -> typing.Union[
        "Port8111.BeanIndicatorAir",
        "Port8111.BeanIndicatorTank",
        "Port8111.BeanMapInfo",
        "Port8111.BeanState",
        "Port8111.BeanInvalid",
    ]:
        json_data = Port8111.get_raw_json(queryType, param=param, timeout=timeout)
        return queryType.parseJson(json_data)


@Singleton
class Port8111Cache:
    typeCache: "dict[Port8111.QueryType, Port8111Cache.SingleTypeCache]"

    class SingleTypeCache(Cache):
        queryType: Port8111.QueryType

        def __init__(self, queryType, fetch8111Interval) -> None:
            super().__init__(
                toFetch=lambda: Port8111.get(queryType),
                updateStrategey=Cache.UpdateStrategey.Outdated(fetch8111Interval),
            )
            self.queryType = queryType

    def __init__(self, fetch8111Interval=None) -> None:
        self.typeCache = dict()
        self.fetch8111Interval = (
            fetch8111Interval if fetch8111Interval is not None else 1
        )

    def get(self, queryType: Port8111.QueryType, newest=None):
        if queryType not in self.typeCache:
            self.typeCache[queryType] = self.SingleTypeCache(
                queryType, self.fetch8111Interval
            )
        return self.typeCache[queryType].get(newest=newest)


@dataclasses.dataclass
class Blkx:

    class FieldType(enum.Enum):
        text = 0
        real = 1
        integer = 2
        bool = 3
        dict = 4

    @dataclasses.dataclass
    class Field:
        name: str
        type: "Blkx.FieldType" = None
        value: typing.Any = None

    fields: "list[Blkx.Field]"

    class _BlkxParser:
        class _TokenType(enum.Enum):
            identifier = 0
            text = 1
            num = 2
            bra = 3
            ket = 4
            colon = 5
            eq = 6
            blank = 7
            eof = 8

        matchers = [
            FSMUtil.RegexpTokenMatcher(
                r"^[a-zA-Z_]{1}[a-zA-Z_0-9]*", _TokenType.identifier
            ),
            FSMUtil.RegexpTokenMatcher(r'^".*"', _TokenType.text),
            FSMUtil.RegexpTokenMatcher(r"^(-)?\d+(\.\d+)", _TokenType.num),
            FSMUtil.RegexpTokenMatcher(r"^{", _TokenType.bra),
            FSMUtil.RegexpTokenMatcher(r"^}", _TokenType.ket),
            FSMUtil.RegexpTokenMatcher(r"^:", _TokenType.colon),
            FSMUtil.RegexpTokenMatcher(r"^=", _TokenType.eq),
            FSMUtil.RegexpTokenMatcher(r"^\s+", _TokenType.blank),
            FSMUtil.RegexpTokenMatcher(r"^$", _TokenType.eof),
        ]

        class _FSMNode(enum.Enum):
            start = 0
            identifier = 1
            colon = 2
            eq = 3
            expectingEq = 4

        @dataclasses.dataclass
        class _ParseReturn:
            value: "list[Blkx.Field]"
            end: int
            endedBy: "Blkx._BlkxParser._TokenType"

        @staticmethod
        def _fields2dict(fields: "list[Blkx.Field]"):
            ret = {}
            for f in fields:
                if f.type == Blkx.FieldType.dict:
                    val = Blkx._BlkxParser._fields2dict(f.value)
                else:
                    val = f.value
                fname = f.name
                if fname not in ret:
                    ret[fname] = val
                else:
                    if not isinstance(ret[fname], list):
                        ret[fname] = [ret[fname]]
                    ret[fname].append(val)
            return ret

        @staticmethod
        def _parseRecursive(s: str, i: int = 0) -> "Blkx._BlkxParser._ParseReturn":
            node = Blkx._BlkxParser._FSMNode.start
            fields = []
            curField = None
            while True:
                while True:
                    token = FSMUtil.getToken(s, i, Blkx._BlkxParser.matchers)
                    i = token.end
                    if token.type != Blkx._BlkxParser._TokenType.blank:
                        break
                if node == Blkx._BlkxParser._FSMNode.start:
                    if token.type == Blkx._BlkxParser._TokenType.identifier:
                        node = Blkx._BlkxParser._FSMNode.identifier
                        curField = Blkx.Field(token.value)
                        fields.append(curField)
                    elif token.type in (
                        Blkx._BlkxParser._TokenType.ket,
                        Blkx._BlkxParser._TokenType.eof,
                    ):
                        # end
                        return Blkx._BlkxParser._ParseReturn(
                            fields, token.end, token.type
                        )
                    else:
                        token.Unexpected()
                elif node == Blkx._BlkxParser._FSMNode.identifier:
                    if token.type == Blkx._BlkxParser._TokenType.bra:
                        curField.type = Blkx.FieldType.dict
                        child = Blkx._BlkxParser._parseRecursive(s, token.end)
                        i = child.end
                        curField.value = child.value
                        node = Blkx._BlkxParser._FSMNode.start
                    elif token.type == Blkx._BlkxParser._TokenType.colon:
                        node = Blkx._BlkxParser._FSMNode.colon
                    else:
                        token.Unexpected()
                elif node == Blkx._BlkxParser._FSMNode.colon:
                    if token.type == Blkx._BlkxParser._TokenType.identifier:
                        if token.value == "t":
                            curField.type = Blkx.FieldType.text
                        elif token.value == "r":
                            curField.type = Blkx.FieldType.real
                        elif token.value == "i":
                            curField.type = Blkx.FieldType.integer
                        elif token.value == "b":
                            curField.type = Blkx.FieldType.bool
                        else:
                            token.Unexpected()
                        node = Blkx._BlkxParser._FSMNode.expectingEq
                    else:
                        token.Unexpected()
                elif node == Blkx._BlkxParser._FSMNode.expectingEq:
                    if token.type == Blkx._BlkxParser._TokenType.eq:
                        node = Blkx._BlkxParser._FSMNode.eq
                    else:
                        token.Unexpected()
                elif node == Blkx._BlkxParser._FSMNode.eq:
                    if token.type in (
                        Blkx._BlkxParser._TokenType.text,
                        Blkx._BlkxParser._TokenType.num,
                        Blkx._BlkxParser._TokenType.identifier,
                    ):
                        value = None
                        if token.type == Blkx._BlkxParser._TokenType.text:
                            # del quotes
                            assert curField.type == Blkx.FieldType.text
                            value = token.value[1:-1]
                        elif token.type == Blkx._BlkxParser._TokenType.num:
                            assert curField.type in (
                                Blkx.FieldType.real,
                                Blkx.FieldType.integer,
                            )
                            if curField.type == Blkx.FieldType.real:
                                value = float(token.value)
                            elif curField.type == Blkx.FieldType.integer:
                                value = int(token.value)
                        elif token.type == Blkx._BlkxParser._TokenType.identifier:
                            if token.value == "yes":
                                value = True
                            elif token.value == "no":
                                value = False
                            else:
                                token.Unexpected()
                            assert curField.type == Blkx.FieldType.bool
                        curField.value = value
                        node = Blkx._BlkxParser._FSMNode.start
                    else:
                        token.Unexpected()

    @staticmethod
    def fromText(s: str):
        return Blkx(Blkx._BlkxParser._parseRecursive(s).value)

    def toDict(self):
        return Blkx._BlkxParser._fields2dict(self.fields)
