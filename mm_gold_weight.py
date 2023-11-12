# TODO: check code | NOT DONE
# TODO: add short doc strings | NOT DONE
# TODO: analyse the logic and finding potential bugs and pitfalls | NOT DONE
# TODO: unit test | NOT DONE
# TODO: benchmark test | NOT DONE
# TODO: writing full documentation | NOT DONE

import decimal
from decimal import Decimal
from operator import eq, ne, gt, lt, ge, le, add, sub, mul, truediv
from types import MappingProxyType
from typing import Iterable, Union


# OPERATORS FUNCTIONS
_COMPARISON_OPERATORS = MappingProxyType({
    "==": eq,
    "!=": ne,
    ">": gt,
    "<": lt,
    ">=": ge,
    "<=": le
})

_ADD_SUB_OPERATORS = MappingProxyType({
    "+": add,
    "-": sub
})

_MUL_DIV_OPERATORS = MappingProxyType({
    "*": mul,
    "/": truediv
})


# LOCAL UNITS AND STANDARDS
_CONVERSION_FACTORS = {
        "MG1": MappingProxyType({
            "ONE_PE_IN_YWAY": Decimal("8"),
            "ONE_KYAT_IN_PE": Decimal("16"),
            "ONE_KYAT_IN_YWAY": Decimal("16") * Decimal("8"),
            "ONE_PEITTHA_IN_KYAT": Decimal("100"),
            "ONE_PEITTHA_IN_PE": Decimal("100") * Decimal("16"),
            "ONE_PEITTHA_IN_YWAY": Decimal("100") * Decimal("16") * Decimal("8")
        }),

        "MG2": MappingProxyType({
            "ONE_PE_IN_YWAY": Decimal("7.5"),
            "ONE_KYAT_IN_PE": Decimal("16"),
            "ONE_KYAT_IN_YWAY": Decimal("16") * Decimal("7.5"),
            "ONE_PEITTHA_IN_KYAT": Decimal("100"),
            "ONE_PEITTHA_IN_PE": Decimal("100") * Decimal("16"),
            "ONE_PEITTHA_IN_YWAY": Decimal("100") * Decimal("16") * Decimal("7.5")
        })
    }

# FOREIGN UNITS
_ONE_KYAT_IN_GRAM_MG1 = Decimal("16.6")
_ONE_OUNCE_IN_GRAM = Decimal("28.3495")
_ONE_GRAM_IN_CARAT = Decimal("5")


def set_kyat_to_gram_conversion(new_value: Union[int, Decimal]):
    if not isinstance(new_value, (int, Decimal)):
        raise TypeError(f"expected type: 'int' or 'Decimal' for new_value"
                        f", but got {type(new_value).__name__!r} instead!")

    if not new_value > 0:
        raise ValueError(f"new_value must be greater than zero!")

    global _ONE_KYAT_IN_GRAM_MG1
    _ONE_KYAT_IN_GRAM_MG1 = Decimal(new_value)


def add_standard(standard: str, one_pe_in_yway: Union[int, Decimal]):
    if not isinstance(standard, str):
        raise TypeError(f"expected type: 'str' for standard"
                        f", but got {type(standard).__name__!r} instead!")

    if not isinstance(one_pe_in_yway, (int, Decimal)):
        raise TypeError(f"expected type: 'int' or 'Decimal' for one_pe_in_yway"
                        f", but got {type(one_pe_in_yway).__name__!r} instead!")

    if standard in _CONVERSION_FACTORS:
        raise ValueError(f"standard {standard!r} already exists!")

    one_pe_in_yway_dec = Decimal(one_pe_in_yway)

    _CONVERSION_FACTORS[standard] = MappingProxyType({
        "ONE_PE_IN_YWAY": one_pe_in_yway_dec,
        "ONE_KYAT_IN_PE": Decimal("16"),
        "ONE_KYAT_IN_YWAY": Decimal("16") * one_pe_in_yway_dec,
        "ONE_PEITTHA_IN_KYAT": Decimal("100"),
        "ONE_PEITTHA_IN_PE": Decimal("100") * Decimal("16"),
        "ONE_PEITTHA_IN_YWAY": Decimal("100") * Decimal("16") * one_pe_in_yway_dec
    })


def remove_standard(standard: str):
    if standard not in _CONVERSION_FACTORS:
        raise ValueError(f"standard {standard!r} does not exist!")

    if standard in ("MG1", "MG2"):
        raise ValueError(f"standard {standard!r} is protected and cannot be removed!")

    del _CONVERSION_FACTORS[standard]


class KPY:

    # Initialization and Representation Methods:

    def __init__(self, kyat: Union[str, int, Decimal] = 0,
                 pe: Union[str, int, Decimal] = 0,
                 yway: Union[str, int, Decimal] = 0,
                 sign: int = 1,
                 standard: str = "MG1"):

        self.kyat = kyat
        self.pe = pe
        self.yway = yway

        self.sign = sign

        self._setter_for_standard_attribute(standard)

    def __repr__(self):
        return (f"KPY(kyat={self.repr_kyat!r}, pe={self.repr_pe!r}, yway={self.repr_yway!r}"
                f", sign={self.sign!r}, standard={self.standard!r})")

    def __str__(self):
        return f"{self._get_sign_str(self.sign)} {self.kyat} Kyat, {self.pe} Pe, {self.yway} Yway ({self.standard})"

    # Class Methods (Public):

    @classmethod
    def summation(cls, weight_instances: Iterable['KPY']) -> 'KPY':
        result_in_yway_unit = 0

        weight_instances_iter = iter(weight_instances)
        first_weight_instance = next(weight_instances_iter)

        if not isinstance(first_weight_instance, KPY):
            raise TypeError(f"expected types: KPY instances or its sub-class instances"
                            f", but found {type(first_weight_instance).__name__!r}!")

        overall_standard = first_weight_instance.standard

        result_in_yway_unit = result_in_yway_unit + first_weight_instance.to_yway_unit()

        for weight_instance in weight_instances_iter:
            if not isinstance(weight_instance, KPY):
                raise TypeError(f"expected types: KPY instances or its sub-class instances"
                                f", but found {type(weight_instance).__name__!r}!")

            if weight_instance.standard != overall_standard:
                raise ValueError(f"all KPY instances or its sub-class instances must use the same standard."
                                 f" expected standard: {overall_standard!r}, but found {weight_instance.standard!r}!")

            result_in_yway_unit = result_in_yway_unit + weight_instance.to_yway_unit()

        result_weight_instance = cls(yway=abs(result_in_yway_unit), sign=cls._get_sign(result_in_yway_unit),
                                     standard=overall_standard)
        result_weight_instance.normalize()
        return result_weight_instance

    # Class Methods (Internal):
    @classmethod
    def _yway_difference_per_kyat_between_standards(cls, standard_1: str, standard_2: str) -> Decimal:
        valid_standards = _CONVERSION_FACTORS.keys()

        if standard_1 not in valid_standards:
            valid_standards_str = map(repr, valid_standards)
            raise ValueError(f"provided standards must be one of these values: {'or '.join(valid_standards_str)}!")

        one_kyat_in_yway_standard_1 = _CONVERSION_FACTORS[standard_1]["ONE_KYAT_IN_YWAY"]
        one_kyat_in_yway_standard_2 = _CONVERSION_FACTORS[standard_2]["ONE_KYAT_IN_YWAY"]

        return one_kyat_in_yway_standard_1 - one_kyat_in_yway_standard_2

    # Properties (Getters and Setters):

    @property
    def kyat(self) -> Decimal:
        return self._kyat

    @property
    def pe(self) -> Decimal:
        return self._pe

    @property
    def yway(self) -> Decimal:
        return self._yway

    @property
    def repr_kyat(self) -> Union[str, int, Decimal]:
        return self._repr_kyat

    @property
    def repr_pe(self) -> Union[str, int, Decimal]:
        return self._repr_pe

    @property
    def repr_yway(self) -> Union[str, int, Decimal]:
        return self._repr_yway

    @property
    def sign(self) -> int:
        return self._sign

    @property
    def standard(self) -> str:
        return self._standard

    @kyat.setter
    def kyat(self, kyat_value: Union[int, str, Decimal]) -> None:
        self._setter_for_weight_attribute("kyat", kyat_value)

    @pe.setter
    def pe(self, pe_value: Union[int, str, Decimal]) -> None:
        self._setter_for_weight_attribute("pe", pe_value)

    @yway.setter
    def yway(self, yway_value: Union[int, str, Decimal]) -> None:
        self._setter_for_weight_attribute("yway", yway_value)

    @sign.setter
    def sign(self, sign_value: int) -> None:
        if not isinstance(sign_value, int):
            raise TypeError(f"expected type: 'int' for sign, but got {type(sign_value).__name__!r} instead!")

        if not(sign_value == -1 or sign_value == 1):
            raise ValueError("sign must be -1 or 1!")

        self._sign = sign_value

    # Instance Methods (Public):

    def is_empty_weight(self) -> bool:
        return all((not self.kyat, not self.pe, not self.yway))

    def is_true_positive_weight(self) -> bool:
        return self.sign == 1 and not self.is_empty_weight()

    def is_true_negative_weight(self) -> bool:
        return self.sign == -1 and not self.is_empty_weight()

    def set_to_zero(self) -> None:
        """Set all weight attributes to zero."""
        self.kyat = 0
        self.pe = 0
        self.yway = 0

    def to_kyat_unit(self) -> Decimal:
        return (self.kyat +
                (self.pe / _CONVERSION_FACTORS[self.standard]["ONE_KYAT_IN_PE"]) +
                (self.yway / _CONVERSION_FACTORS[self.standard]["ONE_KYAT_IN_YWAY"])) * self.sign

    def to_pe_unit(self) -> Decimal:
        return ((self.kyat * _CONVERSION_FACTORS[self.standard]["ONE_KYAT_IN_PE"]) +
                self.pe +
                (self.yway / _CONVERSION_FACTORS[self.standard]["ONE_PE_IN_YWAY"])) * self.sign

    def to_yway_unit(self) -> Decimal:
        return ((self.kyat * _CONVERSION_FACTORS[self.standard]["ONE_KYAT_IN_YWAY"]) +
                (self.pe * _CONVERSION_FACTORS[self.standard]["ONE_PE_IN_YWAY"]) +
                self.yway) * self.sign

    def normalize(self) -> None:
        # Conversation factors
        one_pe_in_yway = Decimal(_CONVERSION_FACTORS[self.standard]["ONE_PE_IN_YWAY"])
        one_kyat_in_yway = Decimal(_CONVERSION_FACTORS[self.standard]["ONE_KYAT_IN_YWAY"])

        # Convert the weight to yway unit (absolute)
        abs_weight_in_yway_unit = abs(self.to_yway_unit())

        # Normalizing
        self.kyat = abs_weight_in_yway_unit // one_kyat_in_yway
        remaining_yway = abs_weight_in_yway_unit % one_kyat_in_yway
        self.pe = remaining_yway // one_pe_in_yway
        self.yway = remaining_yway % one_pe_in_yway

    def change_standard(self, new_standard_value: str) -> None:
        previous_standard_value = self.standard

        if new_standard_value != previous_standard_value:
            # Convert everything to yway unit for standard conversion
            weight_in_yway_unit = self.to_yway_unit()
            weight_in_kyat_unit = self.to_kyat_unit()

            yway_difference = self._yway_difference_per_kyat_between_standards(previous_standard_value,
                                                                               new_standard_value)
            weight_adjustment = weight_in_kyat_unit * yway_difference
            weight_in_yway_unit = weight_in_yway_unit - weight_adjustment

            self._setter_for_standard_attribute(new_standard_value)

            # Update and set the value
            self.set_to_zero()
            self.yway = abs(weight_in_yway_unit)
            self.sign = self._get_sign(weight_in_yway_unit)
            self.normalize()

    # Instance Methods (Internal):

    def _setter_for_weight_attribute(self, attribute_name: str,
                                     value: Union[str, int, Decimal]) -> None:

        internal_attribute_name = f"_{attribute_name}"
        internal_repr_attribute_name = f"_repr_{attribute_name}"

        # Check the value type
        if not isinstance(value, (str, int, Decimal)):
            raise TypeError(f"expected type: 'str' or 'int' or 'Decimal' for {attribute_name}"
                            f", but got {type(value).__name__!r} instead!")

        # Convert value to decimal
        try:
            value_dec = Decimal(value)
        except decimal.InvalidOperation:
            raise ValueError(f"failed to convert to Decimal, {value!r} is a invalid value!") from None

        # Check if it is negative or not
        if not value_dec >= 0:
            raise ValueError(f"{attribute_name} must be a positive value!")

        # Set the attr
        setattr(self, internal_attribute_name, value_dec)
        # Just for the repr
        setattr(self, internal_repr_attribute_name, value)

    def _setter_for_standard_attribute(self, standard_value: str) -> None:
        valid_standards = _CONVERSION_FACTORS.keys()

        if standard_value not in valid_standards:
            valid_standards_str = map(repr, valid_standards)
            raise ValueError(f"standard must be one of these values: {'or '.join(valid_standards_str)}!")

        setattr(self, "_standard", standard_value)

    def _perform_comparison(self, other: Union[tuple, list, 'KPY'], operation: str, *,
                            weight_cls: type) -> bool:

        if isinstance(other, (tuple, list)):
            other = weight_cls(*other)

        if not isinstance(other, KPY):
            return NotImplemented

        if self.standard != other.standard:
            raise ValueError(f"cannot perform {operation!r} between KPY instances or its sub-class instances"
                             f" with different standards: {self.standard!r} and {other.standard!r}!")

        self_in_yway_unit = self.to_yway_unit()
        other_in_yway_unit = other.to_yway_unit()

        com_op_fn = _COMPARISON_OPERATORS[operation]

        return com_op_fn(self_in_yway_unit, other_in_yway_unit)

    def _perform_addition_subtraction(self, other: Union[tuple, list, 'KPY'], operation: str,
                                      reverse: bool = False, in_place: bool = False, *,
                                      weight_cls: type) -> 'KPY':
        if reverse and in_place:
            raise ValueError("Invalid method call: trying to do reverse and in_place at the same time!")

        if isinstance(other, (tuple, list)):
            other = weight_cls(*other)

        if not isinstance(other, KPY):
            return NotImplemented

        if self.standard != other.standard:
            raise ValueError(f"cannot perform {operation!r} between KPY instances or its sub-class instances"
                             f" with different standards: {self.standard!r} and {other.standard!r}!")

        add_sub_op_fn = _ADD_SUB_OPERATORS[operation]

        if reverse:
            result_in_yway_unit = add_sub_op_fn(other.to_yway_unit(), self.to_yway_unit())
        else:
            result_in_yway_unit = add_sub_op_fn(self.to_yway_unit(), other.to_yway_unit())

        if in_place:
            self.set_to_zero()
            self.yway = abs(result_in_yway_unit)
            self.sign = self._get_sign(result_in_yway_unit)
            self.normalize()
            return self
        else:
            result_weight_instance = weight_cls(yway=abs(result_in_yway_unit),
                                                sign=self._get_sign(result_in_yway_unit),
                                                standard=self.standard)
            result_weight_instance.normalize()
            return result_weight_instance

    def _perform_multiplication_division(self, scalar: Union[int, Decimal], operation: str,
                                         in_place: bool = False, *, weight_cls: Union[None, type]) -> 'KPY':

        if not isinstance(scalar, (int, Decimal)):
            return NotImplemented

        mul_div_op_fn = _ADD_SUB_OPERATORS[operation]
        result_in_yway_unit = mul_div_op_fn(self.to_yway_unit(), scalar)

        if in_place:
            self.set_to_zero()
            self.yway = abs(result_in_yway_unit)
            self.sign = self._get_sign(result_in_yway_unit)
            self.normalize()
            return self
        else:
            result_weight_instance = weight_cls(yway=abs(result_in_yway_unit),
                                                sign=self._get_sign(result_in_yway_unit),
                                                standard=self.standard)
            result_weight_instance.normalize()
            return result_weight_instance

    # Static Methods (Public):

    @staticmethod
    def _get_sign(value: Union[int, Decimal]) -> int:
        return 1 if value >= 0 else -1

    @staticmethod
    def _get_sign_str(value: Union[int, Decimal]) -> str:
        return '+' if value >= 0 else '-'

    # Magic Methods:

    def __eq__(self, other):
        return self._perform_comparison(other, "==", weight_cls=KPY)

    def __gt__(self, other):
        return self._perform_comparison(other, ">", weight_cls=KPY)

    def __lt__(self, other):
        return self._perform_comparison(other, "<", weight_cls=KPY)

    def __ne__(self, other):
        return self._perform_comparison(other, "!=", weight_cls=KPY)

    def __ge__(self, other):
        return self._perform_comparison(other, ">=", weight_cls=KPY)

    def __le__(self, other):
        return self._perform_comparison(other, "<=", weight_cls=KPY)

    def __add__(self, other):
        return self._perform_addition_subtraction(other, "+", weight_cls=KPY)

    def __sub__(self, other):
        return self._perform_addition_subtraction(other, "-", weight_cls=KPY)

    def __mul__(self, scalar):
        return self._perform_multiplication_division(scalar, "*", weight_cls=KPY)

    def __truediv__(self, scalar):
        return self._perform_multiplication_division(scalar, "/", weight_cls=KPY)

    def __radd__(self, other):
        # Setting reverse = True does not matter for addition
        return self._perform_addition_subtraction(other, "+", weight_cls=KPY)

    def __rsub__(self, other):
        # Setting reverse = True is matter for subtraction
        return self._perform_addition_subtraction(other, "-", reverse=True, weight_cls=KPY)

    def __rmul__(self, scalar):
        # rmul and mul will behave the same
        return self._perform_multiplication_division(scalar, "*", weight_cls=KPY)

    def __iadd__(self, other):
        return self._perform_addition_subtraction(other, "+", in_place=True, weight_cls=KPY)

    def __isub__(self, other):
        return self._perform_addition_subtraction(other, "-", in_place=True, weight_cls=KPY)

    def __imul__(self, scalar):
        # weight_cls = None because when in_place = True, it does not need to create a new weight instance
        return self._perform_multiplication_division(scalar, "*", in_place=True, weight_cls=None)

    def __itruediv__(self, scalar):
        # weight_cls = None because when in_place = True, it does not need to create a new weight instance
        return self._perform_multiplication_division(scalar, "/", in_place=True, weight_cls=None)

    def __bool__(self):
        return not self.is_empty_weight()


class ForeignKPY(KPY):

    # Representation Method:

    def __repr__(self):
        return (f"ForeignKPY(kyat={self.repr_kyat!r}, pe={self.repr_pe!r}, yway={self.repr_yway!r}"
                f", sign={self.sign!r}, standard={self.standard!r})")

    # Class Methods (Public):

    @classmethod
    def from_gram_unit(cls, gram_value: Union[int, Decimal]) -> 'ForeignKPY':
        # Convert the value to the kyat unit
        weight_in_kyat_unit = gram_value / _ONE_KYAT_IN_GRAM_MG1

        # Create the weight instance attributes using calculated result
        weight_instance = cls(kyat=abs(weight_in_kyat_unit), sign=cls._get_sign(weight_in_kyat_unit), standard="MG1")

        # Normalize and return the weight instance
        weight_instance.normalize()
        return weight_instance

    @classmethod
    def from_carat_unit(cls, carat_value: Union[int, Decimal]) -> 'ForeignKPY':
        carat_to_gram_unit = carat_value / _ONE_GRAM_IN_CARAT
        return cls.from_gram_unit(carat_to_gram_unit)

    @classmethod
    def from_ounce_unit(cls, ounce_value: Union[int, Decimal]) -> 'ForeignKPY':
        ounce_to_gram_unit = ounce_value * _ONE_OUNCE_IN_GRAM
        return cls.from_gram_unit(ounce_to_gram_unit)

    # Instance Methods (Public):

    def to_gram_unit(self) -> Decimal:
        if self.standard != "MG1":
            raise ValueError(f"to_gram_unit method is only applicable for 'MG1' standard!")

        return self.to_kyat_unit() * _ONE_KYAT_IN_GRAM_MG1

    def to_carat_unit(self) -> Decimal:
        return self.to_gram_unit() * _ONE_GRAM_IN_CARAT

    def to_ounce_unit(self) -> Decimal:
        return self.to_gram_unit() / _ONE_OUNCE_IN_GRAM

    # Magic Methods:

    def __add__(self, other):
        return self._perform_addition_subtraction(other, "+", weight_cls=ForeignKPY)

    def __sub__(self, other):
        return self._perform_addition_subtraction(other, "-", weight_cls=ForeignKPY)

    def __mul__(self, scalar):
        return self._perform_multiplication_division(scalar, "*", weight_cls=ForeignKPY)

    def __truediv__(self, scalar):
        return self._perform_multiplication_division(scalar, "/", weight_cls=ForeignKPY)

    def __radd__(self, other):
        # Setting reverse = True does not matter for addition
        return self._perform_addition_subtraction(other, "+", weight_cls=ForeignKPY)

    def __rsub__(self, other):
        # Setting reverse = True is matter for subtraction
        return self._perform_addition_subtraction(other, "-", reverse=True, weight_cls=ForeignKPY)

    def __rmul__(self, scalar):
        # rmul and mul will behave the same
        return self._perform_multiplication_division(scalar, "*", weight_cls=ForeignKPY)

    def __iadd__(self, other):
        return self._perform_addition_subtraction(other, "+", in_place=True, weight_cls=ForeignKPY)

    def __isub__(self, other):
        return self._perform_addition_subtraction(other, "-", in_place=True, weight_cls=ForeignKPY)

    def __imul__(self, scalar):
        # weight_cls = None because when in_place = True, it does not need to create a new weight instance
        return self._perform_multiplication_division(scalar, "*", in_place=True, weight_cls=None)

    def __itruediv__(self, scalar):
        # weight_cls = None because when in_place = True, it does not need to create a new weight instance
        return self._perform_multiplication_division(scalar, "/", in_place=True, weight_cls=None)


class ExtendedForeignKPY(ForeignKPY):

    # Representation Method:

    def __repr__(self):
        return (f"ExtendedForeignKPY(kyat={self.repr_kyat!r}, pe={self.repr_pe!r}, yway={self.repr_yway!r}"
                f", sign={self.sign!r}, standard={self.standard!r})")

    # Instance Methods (Public):

    def calculate_price(self, *,
                        price_per_kyat: Union[int, Decimal] = None,
                        price_per_pe: Union[int, Decimal] = None,
                        price_per_yway: Union[int, Decimal] = None,
                        price_per_gram: Union[int, Decimal] = None,
                        price_per_carat: Union[int, Decimal] = None,
                        price_per_ounce: Union[int, Decimal] = None) -> Decimal:

        # Count how many price arguments are provided
        provided_prices = [arg for arg in (price_per_kyat, price_per_pe, price_per_yway,
                                           price_per_gram, price_per_carat, price_per_ounce) if arg is not None]

        # Validate that only one price argument is provided
        if len(provided_prices) != 1:
            raise ValueError("must provide exactly one price argument!")

        price = provided_prices[0]

        if not price >= 0:
            raise ValueError("price must be a positive value!")

        if self.is_true_negative_weight():
            raise ValueError("weight must be positive value to calculate its price!")

        # Calculate price based on the provided argument
        if price_per_kyat is not None:
            return self.to_kyat_unit() * price
        elif price_per_pe is not None:
            return self.to_pe_unit() * price
        elif price_per_yway is not None:
            return self.to_yway_unit() * price
        elif price_per_gram is not None:
            return self.to_gram_unit() * price
        elif price_per_carat is not None:
            return self.to_carat_unit() * price
        elif price_per_ounce is not None:
            return self.to_ounce_unit() * price

    def calculate_loss_rate(self, new_weight: KPY) -> Decimal:
        if not isinstance(new_weight, KPY):
            raise TypeError(f"expected type: KPY instance or its sub-class instance for new_weight"
                            f", but got {type(new_weight).__name__!r}!")

        if self.is_empty_weight() or self.is_true_negative_weight():
            raise ValueError("original_weight must be non-empty and positive value!")

        if new_weight.is_true_negative_weight():
            raise ValueError("new_weight must be positive value!")

        if self.standard != new_weight.standard:
            raise ValueError(f"cannot calculate loss rate between KPY instances or its sub-class instances"
                             f" with different standards: {self.standard!r} and {new_weight.standard!r}!")

        one_kyat_in_yway = _CONVERSION_FACTORS[self.standard]["ONE_KYAT_IN_YWAY"]

        loss_yway = self.to_yway_unit() - new_weight.to_yway_unit()

        return (loss_yway * one_kyat_in_yway) / self.to_yway_unit()

    # Magic Methods:

    def __add__(self, other):
        return self._perform_addition_subtraction(other, "+", weight_cls=ExtendedForeignKPY)

    def __sub__(self, other):
        return self._perform_addition_subtraction(other, "-", weight_cls=ExtendedForeignKPY)

    def __mul__(self, scalar):
        return self._perform_multiplication_division(scalar, "*", weight_cls=ExtendedForeignKPY)

    def __truediv__(self, scalar):
        return self._perform_multiplication_division(scalar, "/", weight_cls=ExtendedForeignKPY)

    def __radd__(self, other):
        # Setting reverse = True does not matter for addition
        return self._perform_addition_subtraction(other, "+", weight_cls=ExtendedForeignKPY)

    def __rsub__(self, other):
        # Setting reverse = True is matter for subtraction
        return self._perform_addition_subtraction(other, "-", reverse=True, weight_cls=ExtendedForeignKPY)

    def __rmul__(self, scalar):
        # rmul and mul will behave the same
        return self._perform_multiplication_division(scalar, "*", weight_cls=ExtendedForeignKPY)

    def __iadd__(self, other):
        return self._perform_addition_subtraction(other, "+", in_place=True, weight_cls=ExtendedForeignKPY)

    def __isub__(self, other):
        return self._perform_addition_subtraction(other, "-", in_place=True, weight_cls=ExtendedForeignKPY)

    def __imul__(self, scalar):
        # weight_cls = None because when in_place = True, it does not need to create a new weight instance
        return self._perform_multiplication_division(scalar, "*", in_place=True, weight_cls=None)

    def __itruediv__(self, scalar):
        # weight_cls = None because when in_place = True, it does not need to create a new weight instance
        return self._perform_multiplication_division(scalar, "/", in_place=True, weight_cls=None)


class PeitKPY(ExtendedForeignKPY):

    # Initialization and Representation Methods:

    def __init__(self, peittha: Union[str, int, Decimal] = 0,
                 kyat: Union[str, int, Decimal] = 0,
                 pe: Union[str, int, Decimal] = 0,
                 yway: Union[str, int, Decimal] = 0,
                 sign: int = 1,
                 standard: str = "MG1"):

        self.peittha = peittha
        super().__init__(kyat=kyat, pe=pe, yway=yway, sign=sign, standard=standard)

    def __repr__(self):
        return (f"PeitKPY(peittha={self.peittha!r}, kyat={self.kyat!r}, pe={self.pe!r}, yway={self.yway!r},"
                f" sign={self.sign!r}, standard={self.standard!r})")

    def __str__(self):
        return (f"{self._get_sign_str(self.sign)} {self.peittha} Peittha, {self.kyat} Kyat"
                f", {self.pe} Pe, {self.yway} Yway ({self.standard})")

    # Properties (Getters and Setters):

    @property
    def peittha(self) -> Decimal:
        return self._peittha

    @property
    def repr_peittha(self) -> Union[str, int, Decimal]:
        return self._repr_peittha

    @peittha.setter
    def peittha(self, peittha_value) -> None:
        self._setter_for_weight_attribute("peittha", peittha_value)

    # Instance Methods (Public):

    def is_empty_weight(self) -> bool:
        return not self.peittha and super().is_empty_weight()

    def set_to_zero(self) -> None:
        self.peittha = 0
        super().set_to_zero()

    def to_peittha_unit(self) -> Decimal:
        kpy_to_peittha_unit = super().to_kyat_unit() / _CONVERSION_FACTORS[self.standard]["ONE_PEITTHA_IN_KYAT"]
        peittha = self.peittha * self.sign
        return kpy_to_peittha_unit + peittha

    def to_kyat_unit(self) -> Decimal:
        peittha_to_kyat_unit = (self.peittha * _CONVERSION_FACTORS[self.standard]["ONE_PEITTHA_IN_KYAT"]) * self.sign
        return peittha_to_kyat_unit + super().to_kyat_unit()

    def to_pe_unit(self) -> Decimal:
        peittha_to_pe_unit = (self.peittha * _CONVERSION_FACTORS[self.standard]["ONE_PEITTHA_IN_PE"]) * self.sign
        return peittha_to_pe_unit + super().to_pe_unit()

    def to_yway_unit(self) -> Decimal:
        peittha_to_yway_unit = (self.peittha * _CONVERSION_FACTORS[self.standard]["ONE_PEITTHA_IN_YWAY"]) * self.sign
        return peittha_to_yway_unit + super().to_yway_unit()

    def normalize(self) -> None:
        super().normalize()
        one_peittha_in_kyat = _CONVERSION_FACTORS[self.standard]["ONE_PEITTHA_IN_KYAT"]
        abs_kyat = self.kyat  # No need to use abs function because this class only store weight value as abs value
        self.peittha = abs_kyat // one_peittha_in_kyat
        self.kyat = abs_kyat % one_peittha_in_kyat

    # Magic Methods:

    def __eq__(self, other):  # Override comparisons in PeitKPY but not in above subclasses? It is because I added a new
        return self._perform_comparison(other, "==", weight_cls=PeitKPY)  # weight attribute "Peittha" in this class

    def __gt__(self, other):
        return self._perform_comparison(other, ">", weight_cls=PeitKPY)

    def __lt__(self, other):
        return self._perform_comparison(other, "<", weight_cls=PeitKPY)

    def __ne__(self, other):
        return self._perform_comparison(other, "!=", weight_cls=PeitKPY)

    def __ge__(self, other):
        return self._perform_comparison(other, ">=", weight_cls=PeitKPY)

    def __le__(self, other):
        return self._perform_comparison(other, "<=", weight_cls=PeitKPY)

    def __add__(self, other):
        return self._perform_addition_subtraction(other, "+", weight_cls=PeitKPY)

    def __sub__(self, other):
        return self._perform_addition_subtraction(other, "-", weight_cls=PeitKPY)

    def __mul__(self, scalar):
        return self._perform_multiplication_division(scalar, "*", weight_cls=PeitKPY)

    def __truediv__(self, scalar):
        return self._perform_multiplication_division(scalar, "/", weight_cls=PeitKPY)

    def __radd__(self, other):
        # Setting reverse = True does not matter for addition
        return self._perform_addition_subtraction(other, "+", weight_cls=PeitKPY)

    def __rsub__(self, other):
        # Setting reverse = True is matter for subtraction
        return self._perform_addition_subtraction(other, "-", reverse=True, weight_cls=PeitKPY)

    def __rmul__(self, scalar):
        # rmul and mul will behave the same
        return self._perform_multiplication_division(scalar, "*", weight_cls=PeitKPY)

    def __iadd__(self, other):
        return self._perform_addition_subtraction(other, "+", in_place=True, weight_cls=PeitKPY)

    def __isub__(self, other):
        return self._perform_addition_subtraction(other, "-", in_place=True, weight_cls=PeitKPY)

    def __imul__(self, scalar):
        # weight_cls = None because when in_place = True, it does not need to create a new weight instance
        return self._perform_multiplication_division(scalar, "*", in_place=True, weight_cls=None)

    def __itruediv__(self, scalar):
        # weight_cls = None because when in_place = True, it does not need to create a new weight instance
        return self._perform_multiplication_division(scalar, "/", in_place=True, weight_cls=None)
