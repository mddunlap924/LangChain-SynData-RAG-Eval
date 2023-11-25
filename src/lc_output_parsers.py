import re
import json
from typing import Tuple, Any, TypeVar
from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from langchain.output_parsers.prompts import NAIVE_FIX_PROMPT
from langchain.output_parsers import (PydanticOutputParser, OutputFixingParser)
from langchain.pydantic_v1 import BaseModel, Field, validator, ValidationError
from langchain.schema import BaseOutputParser, BasePromptTemplate, OutputParserException
from langchain.schema.language_model import BaseLanguageModel


T = TypeVar("T")


def insert_templates(model_template: str,
                     your_system_message: str,
                     user_message_1: str) -> str:
    """
    Insert the System and User Messages into the Model Prompt Template

    Args:
        model_template (str): Model prompt template (e.g. Llama2-Chat)
        your_system_message (str): System Message with placeholders for examples, etc.
        user_message_1 (str): User message with placeholders for context, etc.

    Returns:
        str: Prompt template with placeholders (context, documents, examples, etc.)
    """
    # Insert system message into model template, then insert human message
    template = model_template.replace('{your_system_message}', your_system_message)
    template = template.replace('{user_message_1}', user_message_1)
    return template


class Llama2QuestionAnswer(BaseModel):
    QUESTION: str = Field(description="Your question generated using only information from the context.")
    ANSWER: str = Field(description="Your answer to the generated question.")

    # You can add custom validation logic easily with Pydantic.
    @validator("QUESTION")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question")
        return field


class QuestionAnswerOutputParser(PydanticOutputParser):
    """Output parser for a question and answer response from Llama2-Chat"""

    def __init__(self, **kwargs) -> None:
        # self.prompt_template = prompt_template       
        super(QuestionAnswerOutputParser, self).__init__(pydantic_object=Llama2QuestionAnswer,
                                                         **kwargs)

    def parse(self, text: str) -> Llama2QuestionAnswer:
        try:
            # Greedy search for 1st json candidate.
            match = re.search(
                r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
            )
            json_str = ""
            if match:
                json_str = match.group()
            json_object = json.loads(json_str, strict=False)
            return self.pydantic_object.parse_obj(json_object)

        except (json.JSONDecodeError, ValidationError) as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {text}. Got: {e}"
            raise OutputParserException(msg, llm_output=text)


    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.schema()

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)
        if 'parser_template' in self._lc_kwargs:
            parser_template = self._lc_kwargs['parser_template']
        else:
            parser_template = PYDANTIC_FORMAT_INSTRUCTIONS
        return parser_template.format(schema=schema_str)


class QAOutputFixingParser(BaseOutputParser[T]):
    """Wraps a parser and tries to fix parsing errors."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    parser: BaseOutputParser[T]
    """The parser to use to parse the output."""
    # Should be an LLMChain but we want to avoid top-level imports from langchain.chains
    retry_chain: Any
    """The LLMChain to use to retry the completion."""
    max_retries: int = 1
    """The maximum number of times to retry the parse."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate = NAIVE_FIX_PROMPT,
        max_retries: int = 1,
    ) -> OutputFixingParser[T]:
        """Create an OutputFixingParser from a language model and a parser.

        Args:
            llm: llm to use for fixing
            parser: parser to use for parsing
            prompt: prompt to use for fixing
            max_retries: Maximum number of retries to parse.

        Returns:
            OutputFixingParser
        """
        from langchain.chains.llm import LLMChain

        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(parser=parser, retry_chain=chain, max_retries=max_retries)

    def parse(self, completion: str) -> T:
        retries = 0

        while retries <= self.max_retries:
            try:
                return self.parser.parse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    return Llama2QuestionAnswer(QUESTION='NULL?', ANSWER='NUll')
                else:
                    retries += 1
                    completion = self.retry_chain.run(
                        instructions=self.parser.get_format_instructions(),
                        completion=completion,
                        error=repr(e),
                    )

        raise OutputParserException("Failed to parse")

    async def aparse(self, completion: str) -> T:
        retries = 0

        while retries <= self.max_retries:
            try:
                return await self.parser.aparse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    return Llama2QuestionAnswer(QUESTION='NULL?', ANSWER='NUll')
                else:
                    retries += 1
                    completion = await self.retry_chain.arun(
                        instructions=self.parser.get_format_instructions(),
                        completion=completion,
                        error=repr(e),
                    )

        raise OutputParserException("Failed to parse")

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "output_fixing"
