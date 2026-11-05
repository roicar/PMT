kernelopts(opaquemodules=false);
(*
  evalb(isConstant(0)=true);
  evalb(isConstant(2)=true);
  evalb(isConstant(x+y^2)=false);

*)
isConstant := proc(p)
  return evalb(nops(indets(p))=0);
end proc:


(*

  evalb(isTriangularSet([], [x, y])=true); 
  evalb(isTriangularSet([0], [x, y])=false);
  evalb(isTriangularSet([2], [x, y])=false);
  evalb(isTriangularSet([x+y,x-y^2], [x, y])=false);
  evalb(isTriangularSet([y, x+y^2], [x, y])=false);
  evalb(isTriangularSet([x+y^2, y], [x, y])=true);



*)
#we assert that it is a triangular set
#and the main variables are sorted in descending order: 
#we assume that T is already in modular form 
#if the base characteristic is not zero. 
isTriangularSet := proc(T, vars)
local newC, newT, res, R, L, s:
  if nops(T)=0 then 
     return true; 
  end if: 
  newC, newT := selectremove(xx->nops(indets(xx))=0, T);
  if nops(newC)<>0 then
     return false;
  elif evalb(not  `subset`(indets(T), {op(vars)})) then
     return false;  
  end if: 
  R := RegularChains:-PolynomialRing(vars); 
  L := map(xx->RegularChains:-MainVariable(xx, R), T);
  L := ListTools:-MakeUnique(L);
  if nops(L)=nops(T) then 
     s := map(xx->ListTools:-Search(xx, vars), L);
     if ListTools:-Sorted(s, `<`) then 
        return true;
     else
        return false; 
     end if:  
  else
     return false;
  end if:
end proc:

(*
evalb(isReduced([], [x, y])=true);
evalb(isReduced([x^2+y], [x, y])=true);
evalb(isReduced([x+y^2, y], [x, y])=false);
evalb(isReduced([x+y, y^2+1], [x,y])=true);
#Assume that T is already a triangular set
isReduced := proc(T, vars)
local res: 
   if nops(T)<=1 then
      return true;  
   end if: 
   for i to nops(T)-1 do 
       res := CPrem(T[i], T[i+1..-1], vars[i+1..-1], 1); 
       if expand(res-T[i])<>0 then 
          return false; 
       end if:
   end do: 
   return true;
end proc:
*)

(*
evalb(isReduced([], [x, y])=true);
evalb(isReduced([x^2+y], [x, y])=true);
evalb(isReduced([x+y^2, y], [x, y])=false);
evalb(isReduced([x+y, y^2+1], [x,y])=true);
#Assume that T is already a triangular set
*)
isReduced := proc(T, vars)
local res, R, i: 
   if nops(T)<=1 then
      return true;  
   end if: 
   R := RegularChains:-PolynomialRing(vars);
   for i to nops(T)-1 do 
       res := isReducedPoly(T[i], T[i+1..-1], R); 
       if evalb(not res) then 
          return false; 
       end if:
   end do: 
   return true;
end proc:
(*
R := PolynomialRing([x, y], 3);
evalb(IsReducedPoly(x, [y], R)=true);
evalb(IsReducedPoly(x+y^2,[2*y+1], R)=false);
assume that the polynomials are moduloed.
*)
isReducedPoly := proc(f, T, R)
local i, v:
  for i to nops(T) do
      v := RegularChains:-MainVariable(T[i], R);
      if degree(f, v)>=degree(T[i],v) then
         return false; 
      end if:
  end do:
  return true:
end proc:
(*
evalb(CPrem(x, [y], [x, y])=x);
evalb(CPrem(x+y^2,[2*y+1],[x, y])=4*x+1);
evalb(CPrem(x+y^2,[2*y+1],[x, y], 1)=4*x+1);
*)
CPrem := proc(f, T, vars, char := 0)
local r, R, i, v: 
  R := RegularChains:-PolynomialRing(vars);
  r := f;
  for i to nops(T) do
      v := RegularChains:-MainVariable(T[i], R);
      if char=0 then
         r := sprem(r, T[i], v); 
      else
         r := Sprem(r, T[i], v) mod char;
      end if:
      #if opt=0 then
      #   r := prem(r, T[i], v); 
      #else
      #   r := sprem(r, T[i], v);
      #end if:
  end do:
  return r:
end proc:
(*
evalb(isAscendingSet([x+y^2, y], [x, y])=false);
evalb(isAscendingSet([x+y, y^2+1], [x,y])=true);
*)
#Assume that T is already a triangular set
isAscendingSet := proc(T, vars)
local res:
  if evalb(not isTriangularSet(T, vars)) then 
     return false; 
  end if:
  return isReduced(T, vars);
end proc:
(*
evalb(isCharSet([y^2+x-1, y*(y^3-2*y+1)], [x+y^2-1, x^2+y-1], [x, y])=true);
evalb(isCharSet([y^2+x-1, y*(y^3-2*y+1)], [x+y^2-1, x+y-1], [x, y])=false);
evalb(isCharSet([y^2+x+6, y*(y^3+5*y+1)], [x+y^2+6, x^2+y+6], [x, y], 7)=true);
evalb(isCharSet([y^2+x+6, y*(y^3+5*y+1)], [x+y^2+6, x+y+6], [x, y], 7)=false);
*)
#assume that F is not empty and does not contain 0
isCharSet := proc(in_T, in_F, vars, char := 0)
local newT, res, F, R, T, G, norm;

  if nops(in_F)=0 then
     error "invalid input";
  end if:

  R := RegularChains:-PolynomialRing(vars, char);

 
  F := map(xx->RegularChains:-TRDmodularize_coefficients(xx, R), in_F);
  if evalb(0 in F) then
     error "invalid input after modulo";
  end if:

  
  if nops(in_T)=0 then 
     return evalb(not PolynomialIdeals:-IsProper(PolynomialIdeals:-PolynomialIdeal(F, char)));
  end if:

  T := map(xx->RegularChains:-TRDmodularize_coefficients(xx, R), in_T);
  res := ListTools:-MakeUnique(expand(T-in_T));
  if nops(res)<>1 or res[1]<>0 then
     error "the triangular set is not yet moduloed";
  end if:


  if evalb(not isAscendingSet(T, vars)) then
     return false;
  else
     res := map(xx->CPrem(xx, T, vars, char), F); 
     res := ListTools:-MakeUnique(res); 
     if nops(res)=1 and res[1]=0 then
         G := Groebner:-Basis(F, tdeg(op(vars)), characteristic=char);
         norm := Groebner:-NormalForm(T, G, tdeg(op(vars)), characteristic=char);
         norm := ListTools:-MakeUnique(norm);
         if nops(norm)=1 and norm[1]=0 then
            return true; 
         else
            return false;
         end if:
     else
        return false; 
     end if:
  end if
end proc:
