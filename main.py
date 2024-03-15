# As uno = XI + C
# As de = Anagnosi
# As tre = uno + de
# Grafo tre
import dataclasses
import re
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import NamedTuple, Literal

# Keyword - As|Anagnosi|Grafo
# Variable - [a-z]+
# Number - ^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$
# Arithmetic expression - +|-|*|/
# Variable assignment - =


token_types = [
    (r" +", "WHITESPACE"),
    (r"\n", "EOL"),
    (r"As|Anagnosi|Grafo", "KEYWORD"),
    (r"[a-z]+", "VARIABLE"),
    (r"M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})", "NUMBER"),
    (r"\+|-|\*|/", "ARITHMETIC"),
    (r"=", "ASSIGNMENT"),
]


class Token(NamedTuple):
    token_type: str
    value: str
    position: tuple[int, int]


def tokenize(source: str) -> list[Token]:
    position = 0
    tokens = []

    while position < len(source):
        sub_source = source[position:]
        found_token = False
        for token_regex, token_type in token_types:
            match = re.match(token_regex, sub_source)
            if match is not None and match.span() != (0, 0):
                value = match.group(0)
                start = position + match.span()[0]
                end = start + match.span()[1]
                token = Token(token_type, value, (start, end))
                tokens.append(token)

                start, end = match.span()
                position += end - start
                found_token = True
                break

        if not found_token:
            raise ValueError(f"Invalid expression: {sub_source}")
    return tokens


def format_as_html(tokens: list[Token]) -> str:
    keyword_color = "orange"
    number_color = "blue"

    formatted_source = ""
    for token in tokens:
        if token.token_type == "KEYWORD":
            formatted_token = (
                f'<span style="color:{keyword_color};">{token.value}</span>'
            )
        elif token.token_type == "NUMBER":
            formatted_token = (
                f'<span style="color:{number_color};">{token.value}</span>'
            )
        else:
            formatted_token = token.value
        formatted_source += formatted_token

    return f"<pre>{formatted_source}</pre>"


@dataclasses.dataclass
class Variable:
    name: str


@dataclasses.dataclass
class Number:
    value: int


@dataclasses.dataclass
class ArithmeticAction:
    action: Literal["+", "-", "*", "/"]


@dataclasses.dataclass
class ArithmeticExpression:
    action: ArithmeticAction
    arguments: list["Expression"]


@dataclasses.dataclass
class InputExpression:
    pass


Expression = Variable | Number | ArithmeticExpression | InputExpression


@dataclasses.dataclass
class PrintExpression:
    expression: str


@dataclasses.dataclass
class AssignVariable:
    variable_name: str
    expression: Expression


Statement = AssignVariable | PrintExpression


@dataclasses.dataclass
class AST:
    statements: list[Statement]


def roman_to_int(s: str) -> int:
    sign = -1 if s[0] == "-" else 1

    d = {"m": 1000, "d": 500, "c": 100, "l": 50, "x": 10, "v": 5, "i": 1}
    n = [d[i] for i in s.lower() if i in d]
    return sign * sum(
        [i if i >= n[min(j + 1, len(n) - 1)] else -i for j, i in enumerate(n)]
    )


num_map = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def int_to_roman(num: int) -> str:
    sign = "-" if num < 0 else ""
    num = abs(num)

    roman = ""

    while num > 0:
        for i, r in num_map:
            while num >= i:
                roman += r
                num -= i

    return sign + roman


OPERATORS = {"+": 1, "-": 1, "*": 2, "/": 2}


def parse_arithmetic(tokens: list[Token]) -> Expression:
    # Infix to reverse polish notation using shunting yard algorithm
    operator_stack = []
    rpn = []
    for token in tokens:
        match token:
            case (
                Token(token_type="NUMBER")
                | Token(token_type="KEYWORD", value="Anagnosi")
                | Token(token_type="VARIABLE")
            ):
                rpn.append(token)
            case Token(token_type="ARITHMETIC", value=action):
                while (
                    operator_stack
                    and OPERATORS[operator_stack[-1].value] > OPERATORS[token.value]
                ):
                    rpn.append(operator_stack.pop())
                operator_stack.append(token)
    while operator_stack:
        rpn.append(operator_stack.pop())

    # RPN to Expression
    stack = []
    for token in rpn:
        match token:
            case Token(token_type="ARITHMETIC", value=action):
                arg2, arg1 = stack.pop(), stack.pop()
                stack.append(ArithmeticExpression(action, arguments=[arg1, arg2]))
            case _:
                expression = parse_expression_ast([token])
                stack.append(expression)
    return stack[0]


def parse_expression_ast(tokens: list[Token]) -> Expression:
    match tokens:
        case [Token(token_type="NUMBER", value=roman_number)]:
            integer = roman_to_int(roman_number)
            return Number(integer)
        case [Token(token_type="VARIABLE", value=variable_name)]:
            return Variable(variable_name)
        case [Token(token_type="KEYWORD", value="Anagnosi")]:
            return InputExpression()
        case _:
            return parse_arithmetic(tokens)


def parse_ast(tokens: list[Token]) -> AST:
    # remove whitespaces
    tokens = [token for token in tokens if token.token_type != "WHITESPACE"]
    # break by EOL
    lines = []
    acc_line = []
    for token in tokens:
        if token.token_type == "EOL":
            lines.append(acc_line)
            acc_line = []
        else:
            acc_line.append(token)
    if acc_line:
        lines.append(acc_line)

    statements = []
    for line in lines:
        match line:
            case []:
                pass
            case [
                Token(token_type="KEYWORD", value="As"),
                Token(token_type="VARIABLE", value=variable_name),
                Token(token_type="ASSIGNMENT", value="="),
                *expression_tokens,
            ]:
                expression = parse_expression_ast(expression_tokens)
                statements.append(AssignVariable(variable_name, expression))
            case [Token(token_type="KEYWORD", value="Grafo"), *expression_tokens]:
                expression = parse_expression_ast(expression_tokens)
                statements.append(PrintExpression(expression))
            case _:
                raise ValueError

    ast = AST(statements=statements)
    return ast


def calculate_value(state: dict[str, int], expression: Expression) -> int:
    match expression:
        case Variable(variable_name):
            variable_value = state.get(variable_name)
            if variable_value is None:
                raise ValueError(f"Use variable before assignment: {variable_name}")
            return variable_value
        case Number(variable_value):
            return variable_value
        case ArithmeticExpression(action, arguments):
            values = [calculate_value(state, expression) for expression in arguments]
            first_value, second_value = values
            match action:
                case "+":
                    return first_value + second_value
                case "-":
                    return first_value - second_value
                case "*":
                    return first_value * second_value
                case "/":
                    return first_value // second_value
                case _:
                    raise ValueError(f"Unknown arithmetic action: {action}")
        case InputExpression():
            value_raw = input("Input value (roman integer): ").strip()
            value = roman_to_int(value_raw)
            return value
        case _:
            raise ValueError(f"Unknown expression: {expression}")


def execute(ast: AST) -> None:
    state: dict[str, int] = {}

    for statement in ast.statements:
        match statement:
            case AssignVariable(variable_name, expression):
                value = calculate_value(state, expression)
                state[variable_name] = value
            case PrintExpression(expression):
                value = calculate_value(state, expression)
                roman_value = int_to_roman(value)
                print(f"Value: {roman_value} ({value})")
            case _:
                raise ValueError(f"Unknown statement {statement}")


@dataclasses.dataclass
class LlvmIrCompilerState:
    _vars: dict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: 0)
    )
    _tmp_var_index: int = 0

    def get_next_var(self, var: str) -> str:
        next_value = self._vars[var]
        self._vars[var] += 1
        return f"%{var}.{next_value}"

    def get_previous_var(self, var: str) -> str:
        return f"%{var}.{self._vars[var] - 1}"

    def get_next_tmp_var(self) -> str:
        next_value = self._tmp_var_index
        self._tmp_var_index += 1
        return f"%tmp.{next_value}"

    def get_previous_tmp_var(self) -> str:
        return f"%tmp.{self._tmp_var_index - 1}"


def build_llvm_ir_expression(
    state: LlvmIrCompilerState, expression: Expression
) -> list[str]:
    instructions = []
    match expression:
        case Variable(variable_name):
            instructions.append(
                f"{state.get_next_tmp_var()} = add i32 0, {state.get_previous_var(variable_name)}"
            )
        case Number(number_value):
            instructions.append(
                f"{state.get_next_tmp_var()} = add i32 0, {number_value}"
            )
        case ArithmeticExpression(action, arguments):
            arguments_tmp_var_indexes = []
            for argument in arguments:
                argument_instructions = build_llvm_ir_expression(state, argument)
                arguments_tmp_var_indexes.append(state.get_previous_tmp_var())
                instructions.extend(argument_instructions)

            match action:
                case "+":
                    instruction_opcode = "add"
                case "-":
                    instruction_opcode = "sub"
                case "*":
                    instruction_opcode = "mul"
                case "/":
                    instruction_opcode = "sdiv"
                case _:
                    raise ValueError(f"Unknown action: {action}")
            index1, index2 = arguments_tmp_var_indexes
            instruction = f"{state.get_next_tmp_var()} = {instruction_opcode} i32 {index1}, {index2}"
            instructions.append(instruction)
        case InputExpression():
            result_addr = state.get_next_tmp_var()
            instructions.extend([
                f'{result_addr} = alloca i32, align 4',
                f'store i32 0, ptr {result_addr}, align 4',
                f'call i32 (ptr, ...) @__isoc99_scanf(ptr noundef @scanFormat, ptr noundef {result_addr})',
                f'{state.get_next_tmp_var()} = load i32, ptr {result_addr}, align 4',
            ])
        case _:
            raise ValueError(f"Unknown expression: {expression}")
    return instructions


def build_llvm_ir(ast: AST) -> str:
    state = LlvmIrCompilerState()

    instructions = []
    for statement in ast.statements:
        match statement:
            case AssignVariable(variable_name, expression):
                exp_instructions = build_llvm_ir_expression(state, expression)
                exp_result_index = state.get_previous_tmp_var()
                instruction = f"{state.get_next_var(variable_name)} = add i32 0, {exp_result_index}"

                instructions.extend(exp_instructions)
                instructions.append(instruction)
            case PrintExpression(expression):
                exp_instructions = build_llvm_ir_expression(state, expression)
                exp_result_index = state.get_previous_tmp_var()
                instruction = f"{state.get_next_tmp_var()} = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @formatString , i32 0, i32 0), i32 {exp_result_index})"

                instructions.extend(exp_instructions)
                instructions.append(instruction)
    instructions.append("ret i32 0")

    result = ""
    result += r'@formatString = private constant [4 x i8] c"%d\0A\00"' + "\n"
    result += "declare i32 @printf(ptr noundef, ...)\n"
    result += r'@scanFormat = private unnamed_addr constant [3 x i8] c"%d\00", align 1' + '\n'
    result += "declare i32 @__isoc99_scanf(ptr noundef, ...)\n"
    result += "define i32 @main() {\n"
    for instruction in instructions:
        result += "  " + instruction + "\n"
    result += "}\n"
    return result


source = """
As uno = I + II*III
As de = Anagnosi
As tre = uno + de
Grafo tre
Grafo tre / X
""".strip()
print("Source:")
print(source)
print()

print("Parsing source to tokens")
tokens = tokenize(source)
pprint(tokens)
print()

print("Export as highlighted code to output.html")
html = format_as_html(tokens)
print(html)
Path("output.html").write_text(html)
print()

print("Parsing AST")
ast = parse_ast(tokens)
pprint(ast)
print()

# print("Executing AST (via interpreter)")
# execute(ast)
# print()

print("Compiling to LLVM IR")
llvm_ir = build_llvm_ir(ast)
Path("output.ll").write_text(llvm_ir)

print("Done!")
