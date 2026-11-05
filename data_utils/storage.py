'''
    Utilities for data storage\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 14 Feb 2026
'''
from typing import List, Optional
from torch import Tensor
from torch_geometric.data import HeteroData

class MulData:
    r'''Class storing polynomial multiplication sample.'''

    def __init__(self, system: List, product: str, collected_product: Optional[str] = None) -> None:
        r'''Initialization of the class.

        Args:
            system (List): System of two polynomials.
            product (str): Final product.
            collected_product (Optional[str]): Product with coefficients of like terms collected (default: None).

        Example:
            system: ['5*x^3+3*x^2-4*x-8', '2*x^3-6*x^2-4*x-5']
            product: '10*x^6-24*x^5-46*x^4-29*x^349*x^2+52*x+40'
            collected_product: '+10*x^6(-30+6)*x^5(-20-18-8)*x^4(-25-12+24-16)*x^3(-15+16+48)*x^2(+20+32)*x^1+40'
        '''
        self.system = system
        self.product = product
        self.collected_product = collected_product
        
class AddData:
    r'''Class storing polynomial addition sample.'''

    def __init__(self, system: List, collected_sum: str, sum: str) -> None:
        r'''Initialization of the class.

        Args:
            system (List): System of two polynomials.
            collected_sum (str): Sum with coefficients of like terms collected.
            sum (str): Final sum.
        '''
        self.system = system
        self.collected_sum = collected_sum
        self.sum = sum
        
class LongMulData:
    r'''Class storing long polynomial multiplication sample.'''
    def __init__(self, system: List, product: str, collected_product: List, combined_product: List, collected_sum: List) -> None:
        r'''Initialization of the class.

        Args:
            system (List): System of two polynomials.
            product (str): Final product.
            collected_product (Optional[str]): Product with coefficients of like terms collected (default: None).

        Example:
            
            system: [['5*x^15+3*x^14-4*x^13-8*x^12+2*x^11-6*x^10-4*x^9-5*x^8', '7*x^7-2*x^6-5*x^5-8*x^4+8*x^3+x^2+9*x+4'], ['4*x^15+5*x^14+5*x^13+9*x^12+2*x^11+9*x^10-2*x^9-7*x^8', '7*x^7+7*x^6-7*x^5-5*x^4-5*x^3-8']]
            product: '20*x^30+37*x^29+24*x^28+8*x^27-15*x^26-39*x^25-99*x^24-145*x^23-117*x^22+72*x^21-92*x^20-182*x^19-86*x^18+89*x^17-22*x^16+x^15+80*x^14+378*x^13+114*x^12-72*x^11+69*x^10+115*x^9+91*x^8+28*x^7-64*x^6-38*x^5-x^4-84*x^3-8*x^2-72*x-32'
            collected_product: '+10*x^6(-30+6)*x^5(-20-18-8)*x^4(-25-12+24-16)*x^3(-15+16+48)*x^2(+20+32)*x^1+40'
            collected_product: ['', '', '', '']
            combined_product: [['', ''], ['', ''], ['', '']]
            collected_sum: 
        '''
        self.system = system
        self.product = product
        self.collected_product = collected_product
        self.combined_product = combined_product
        self.collected_sum = collected_sum


class PremData:
    r'''Class storing polynomial pseudodivision sample.'''

    def __init__(self, system: List, one_spa: Optional[List], multi_spa:Optional[List] = None) -> None:
        r'''Initialization of the class.
        
        Args:
            system (List): A system of two polynomials.
            one_spa (Optional[List]): Scratchpad for one-step pseudodivision.
            multi_spa (Optional[List]): Scratchpad for multi-step pseudodivision (default: None).
            
        Example (univariate #1 (coefficients in integer ring)):
            system: ['2*x^3-x^2+2*x-7', '7*x^3-9*x^2-9*x-3']
            one_spa: ['7', '14*x^3-7*x^2+14*x-49', '2', '14*x^3-18*x^2-18*x-6', '11*x^2+32*x-43']
            multi_spa: None
            
        Example (univaraite #2 (coefficients in integer ring)):
            system: ['3*x^4-6*x^3-3*x^2-2*x-9', '-2*x^2-x-2']
            one_spa: ['-8', '12*x^2-30*x-9', '-53*x+54']
            multi_spa: ['-8', '-24*x^4+48*x^3+24*x^2+16*x+72', '12*x^2', '-24*x^4-12*x^3-24*x^2', '60*x^3+48*x^2+16*x+72', '-30*x', '60*x^3+30*x^2+60*x', '18*x^2-44*x+72', '-9', '18*x^2+9*x+18', '-53*x+54']
        
        Example (univariate #3 (coefficients in finite field (F7 for an example))):
            system: ['4*x^7+5*x^6+3*x^4+3*x^3+3*x^2+x+3', '5*x^7+2*x^6+4*x^5+6*x^4+4*x+2']
            one_spa: ['5', '6*x^7+4*x^6+x^4+x^3+x^2+5*x+1', '4', '6*x^7+x^6+2*x^5+3*x^4+2*x+1', '3*x^6+5*x^5+5*x^4+x^3+x^2+3*x']
            multi_spa: None
            
        Example (univariate #4 (coefficients in finite field (F7 for an example))):
            system: ['5*x^5+3*x^3+3*x^2+3*x+1', '5*x^4+2*x^3+4*x^2+6*x']
            one_spa: None
            multi_spa: ['4', '6*x^5+5*x^3+5*x^2+5*x+4', '4*x', '6*x^5+x^4+2*x^3+3*x^2', '6*x^4+3*x^3+2*x^2+5*x+4', '4', '6*x^4+x^3+2*x^2+3*x', '2*x^3+2*x+4']
            
        Example (multivariate #1 (coefficients in integer ring)):
            system: ['-4*x0^5-x0*x1', '7*x0^3*x1^8-x0^4*x1^4']
            one_spa: ['x1^8', '28*x1^8+4*x0*x1^4', '-196*x0^3*x1^16-x0*x1^9']
            multi_spa: ['x1^8', '-4*x0^5*x1^8-x0*x1^9', '4*x1^4*x0', '28*x0^4*x1^12-4*x0^5*x1^8', '-28*x0^4*x1^12-x0*x1^9', '28*x1^8', '196*x0^3*x1^16-28*x0^4*x1^12', '-196*x0^3*x1^16-x0*x1^9']
        
        
        '''
        self.system = system
        self.one_spa = one_spa
        self.multi_spa = multi_spa
        
class CharSetData:
    r'''Class storing characteristic set sample.'''
    
    def __init__(self, system: List, charset: List)->None:
        r'''Initialization of the class.'''
        self.system = system
        self.charset = charset


class SequenceData:
    r'''Sequence to sequence data pair for encoder-decoder architecture or sequence data for decoder-only architecture.'''

    def __init__(self, in_vocabid: Tensor, out_vocabid: Tensor) -> None:
        r'''Initialization of data pair.

        Args:
            in_vocabid (Tensor): Vocabulary indices of tokens in an input/prompt sequence.
            out_vocabid (Tensor): Vocabulary indices of tokens in an output/response sequence.
        '''
        self.in_vocabid = in_vocabid
        self.out_vocabid = out_vocabid


class PosCpData:
    r'''Sequence data for integer accumulation with multi-level position coupling and scratchpad.'''

    def __init__(self, operand: Tensor, operand_id1: Tensor, operand_id2: Tensor, result: Tensor, result_id1: Tensor, result_id2: Tensor, operand_rmov: Tensor) -> None:
        r'''Initialization of sequence data.

        Args:
            operand (Tensor): Vocabulary indices of tokens in integer accumulation formula.
            operand_id1 (Tensor): PosID1 (digit coupling) of the formula.
            operand_id2 (Tensor): PosID2 (operand coupling) of the formula.
            result (Tensor): Vocabular indices of tokens in intermediate addition result(s) and final accumulation result.
            result_id1 (Tensor): PosID1 of the results.
            result_id2 (Tensor): PosID2 of the results.
            operand_rmov (Tensor): Formula >> 1.
        '''
        self.operand = operand
        self.operand_id1 = operand_id1
        self.operand_id2 = operand_id2
        self.result = result
        self.result_id1 = result_id1
        self.result_id2 = result_id2
        self.operand_rmov = operand_rmov
        
class LongSequenceMulData:
    r'''Class storing long sequence polynomial multiplication sample.'''
    def __init__(self, f0g0: Tensor, f0g1:Tensor, f1g0:Tensor, f1g1:Tensor, product: Tensor)->None:
        r'''Initialization of the class.

        Args:
            f0g0 (Tensor): Vocabulary indices of tokens in input f0 * g0.
                    f0g1 (Tensor): Vocabulary indices of tokens in input f0 * g1.
                    f1g0 (Tensor): Vocabulary indices of tokens in input f1 * g0.
                    f1g1 (Tensor): Vocabulary indices of tokens in input f1 * g1.
            product (Tensor): Vocabulary indices of tokens in final product.
        '''
        self.f0g0 = f0g0
        self.f0g1 = f0g1
        self.f1g0 = f1g0
        self.f1g1 = f1g1
        self.product = product
        
class PolyGraphSeq:
    def __init__(self, in_graph: HeteroData, out_seq: Tensor):
        self.in_graph = in_graph
        self.out_seq = out_seq
