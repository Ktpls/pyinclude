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

    class QueryType(enum.Enum):
        indicator = 0
        map_info = 1
        state = 2

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
            Port8111.QueryType.__throwEnumNotFound()

    @staticmethod
    def get_raw_json(queryType: "Port8111.QueryType", timeout=None):
        try:
            response = requests.get(
                "http://127.0.0.1:8111/" + queryType.getPath(),
                timeout=timeout if timeout is not None else 1,
            )
        except (requests.ConnectionError, requests.ReadTimeout):
            return None
        json_data = response.json()
        return json_data

    @staticmethod
    def get(queryType: "Port8111.QueryType", timeout=None):
        json_data = Port8111.get_raw_json(queryType, timeout=timeout)
        return queryType.parseJson(json_data)
