from .roberta.modeling_roberta import RobertaForSequenceClassification 
from .roberta.modeling_roberta import RobertaForMaskedLM

class RobertaForPrompt(RobertaForMaskedLM):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        parser.add_argument("--init_answer_words", type=int, default=1, )
        parser.add_argument("--init_type_words", type=int, default=1, )
        parser.add_argument("--init_answer_words_by_one_token", type=int, default=0, )
        parser.add_argument("--use_template_words", type=int, default=1, )
        return parser

class RobertaForFinetune(RobertaForSequenceClassification):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=False)
        parser.add_argument("--init_answer_words", type=int, default=1)
        parser.add_argument("--init_type_words", type=int, default=0)
        parser.add_argument("--init_answer_words_by_one_token", type=int, default=0)
        parser.add_argument("--use_template_words", type=int, default=0)
        return parser