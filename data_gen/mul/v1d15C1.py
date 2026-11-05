'''
    Generate multiplication dataset 'v1d15C1'\\
    Coded and annotated by Yuxuan SONG
    Last modified time: 13 Feb 2026
'''
from typing import List
from subprocess import check_output, CalledProcessError
from os import remove
from argparse import ArgumentParser
from os.path import join, abspath, dirname
import platform
import sys  # noqa
sys.path.append(join(abspath(dirname(__file__)), '..', '..'))  # noqa
from utils.folder import generate_folder  # noqa
from data_utils.storage import LongMulData  # noqa
from torch import save


def run_maple_with_param(N: int) -> List:
    maple_script = f'''
with(StringTools):

LegalCheck := proc(f::polynom, x::name)
    if degree(f, x) = 15 then
        return true:
    end if:
    return false:
end proc:

RandPoly := proc(x::name)
    local f:
    while true do
        f := randpoly(x, coeffs=rand(-9..9), dense, degree=15):
        if LegalCheck(f, x) then
            break:
        end if:
    end do:
    return f:
end proc:

ObtainTerms := proc(f::polynom)
    if not type(f, `+`) then
        return [f]:
    end if:
    return [op(f)]:
end proc:

ProductTable := proc(f::polynom, g::polynom, x::name)
    local T, ft, pt, exp, c:
    T := table():
    for ft in ObtainTerms(f) do
        for pt in ObtainTerms(expand(ft * g)) do
            exp := degree(pt, x):
            c := coeff(pt, x, exp):
            if not assigned(T[exp]) then
                T[exp] := []:
            end if:
            T[exp] := [op(T[exp]), c]:
        end do:
    end do:
    return T:
end proc:

AddTable := proc(f::polynom, g::polynom, x::name)
    local T, t, exp, c:
    T := table():
    for t in ObtainTerms(f) do
        exp := degree(t, x):
        c := coeff(t, x, exp):
        if not assigned(T[exp]) then
            T[exp] := []:
        end if:
        T[exp] := [op(T[exp]), c]:
    end do:
    for t in ObtainTerms(g) do
        exp := degree(t, x):
        c := coeff(t, x, exp):
        if not assigned(T[exp]) then
            T[exp] := []:
        end if:
        T[exp] := [op(T[exp]), c]:
    end do:
    return T:
end proc:

Format := proc(T::table, x::name)
    local terms, deg, i, like_term, coeffs, term, j:
    terms := []:
    deg := max(indices(T)):
    for i from deg by -1 to 0 do
        if assigned(T[i]) then
            like_term := T[i]:
            if nops(like_term) = 1 then
                # Explicit positive sign
                coeffs := cat(
                    if like_term[1] >= 0 then "+" else "" fi,
                    convert(like_term[1], string)
                ):
                term := coeffs:
                if i >= 1 then
                    term := ""||term||"*x^"||i:
                end if:
            else
                coeffs := "":
                for j to nops(like_term) do
                    coeffs := cat(
                        coeffs,
                        if like_term[j] >= 0 then "+" else "" fi,
                        convert(like_term[j], string)
                    ):
                end do:
                term := "("||coeffs||")*x^"||i:
            end if:
            if nops(terms) = 1 then
                terms := [terms, term]:
            else
                terms := [op(terms), term]:
            end if:
        end if:
    end do:
    if nops(terms) = 1 then
        return terms:
    end if:
    return cat(op(terms)):
end proc:

CollectedProduct := proc(f::polynom, g::polynom, x::name)
    return Format(ProductTable(f, g, x), x)
end proc:

CollectedSum := proc(f::polynom, g::polynom, x::name)
    return Format(AddTable(f, g, x), x)
end proc:

Duplicated := proc(sys::list, L::list)
    local i:
    for i to nops(L) do
        if sys = L[i] then
            return true:
        end if:
    end do:
    return false:
end proc:

ConstructSlice := proc(f::polynom, x::name, max_deg::integer, min_deg::integer)
    local i:
    add(coeff(f, x, i) * x ^ i, i=max_deg..min_deg):
end proc:

SlicePoly := proc(f::polynom, x::name)
    local i:
    return [seq(ConstructSlice(f, x, max(i - 7, 0), i), i = degree(f)..0, -8)]:
end proc:

ConstructSlicedData := proc(f::polynom, g::polynom, x::name)
    local collected_prod, combined_prod, f_list, g_list, i, j, slice_to_be_add, collected_sum, p12_p21, p12_p21_p22:
    collected_prod := "[":
    combined_prod := []:
    f_list := SlicePoly(f, x):
    g_list := SlicePoly(g, x):
    print(cat("[['", cat(convert(f_list[1], string), cat("','", cat(convert(f_list[2], string), cat("'],['", cat(convert(g_list[1], string), cat("','", cat(convert(g_list[2], string), "']]")))))))));
    print("#");
    print(cat("'", cat(expand(f * g), "'")));
    print("#");

    for i to nops(f_list) do
        for j to nops(g_list) do
            collected_prod := cat(collected_prod, cat("'", cat(CollectedProduct(f_list[i], g_list[j], x), "',"))):
            if nops(combined_prod) = 1 then
                combined_prod := [combined_prod[1], expand(f_list[i] * g_list[j])]:
            else
                combined_prod := [op(combined_prod), expand(f_list[i] * g_list[j])]:
            end if:
        end do:
    end do:
    collected_prod := cat(substring(collected_prod, 1..-2), "]"):
    print(collected_prod);
    print("#");

    # p11 + p12
    slice_to_be_add := cat("[['", cat(convert(combined_prod[2], string), cat("','", cat(convert(combined_prod[3], string), "'],['")))):
    collected_sum := cat("['", cat(CollectedSum(combined_prod[2], combined_prod[3], x), "','")):
    p12_p21 := expand(combined_prod[2] + combined_prod[3]):
    # (p12+p21) + p11
    slice_to_be_add := cat(slice_to_be_add, cat(convert(p12_p21, string), cat("','", cat(convert(combined_prod[4], string), "'],['")))):
    collected_sum := cat(collected_sum, cat(CollectedSum(p12_p21, combined_prod[4], x), "','")):
    p12_p21_p22 := expand(p12_p21 + combined_prod[4]):
    # p11 + ((p12+p21)+p22)
    slice_to_be_add := cat(slice_to_be_add, cat(convert(combined_prod[1], string), cat("','", cat(convert(p12_p21_p22, string), "']]")))):
    collected_sum := cat(collected_sum, cat(CollectedSum(combined_prod[1], p12_p21_p22, x), "']")):
    
    print(slice_to_be_add);
    print("#");
    print(collected_sum);
    print("#");
end proc:

systems := []:
GenerateSample := proc()
    local f, x, g, sys:
    global systems:
    while true do
        f := RandPoly(x):
        g := RandPoly(x):
        sys := sort([f, g]):
        if not Duplicated(sys, systems) then
            systems := [op(systems), sys]:
            break:
        end if:
    end do:
    ConstructSlicedData(f, g, x):
    return:
end proc:

randomize(0):
interface(prettyprint = 0):
for i to {N} do
    GenerateSample();
end do:
    '''
    mpl_file_path = join(abspath(dirname(__file__)), 'v1d15C1.mpl')
    with open(mpl_file_path, 'w', encoding='utf-8') as file:
        file.write(maple_script)

    try:
        executor = 'cmaple' if platform.system() == 'Windows' else 'maple'
        command = f'{executor} -q {mpl_file_path}'
        output = check_output(command, shell=True, text=True)
        remove(mpl_file_path)

        muldata = output.strip().replace(' ', '').replace('\n', '').replace(
            '\\', '').replace('"', '').replace(' ', '').split('#')[:-1]
        return muldata

    except CalledProcessError as err:
        raise NotImplementedError(f'Error running Maple: {err}')


parser = ArgumentParser()
parser.add_argument('--N', type=int, default=100000,
                    help='dataset size (default: 100,000)')
args = parser.parse_args()
base_dir = join(abspath(dirname(__file__)), '..', '..', '..',
                'dataset', 'mul', 'v1d15C1')
generate_folder(base_dir)

data = run_maple_with_param(args.N)
for i in range(0, args.N * 5, 5):
    sample = LongMulData(eval(data[i]), eval(
        data[i + 1]), eval(data[i + 2]), eval(data[i + 3]), eval(data[i + 4]))
    save(sample, join(base_dir, str(i // 5) + '.pt'))
