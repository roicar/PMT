# Judge whether a predicted set is a characteristic set of a system of multivariate polynomials
# Coded and annotated by Changbo CHEN and Yuxuan SONG
# Last modified time: 13 Jan 2026
with(ListTools):
with(RegularChains):
with(PolynomialIdeals):
with(Groebner):

isTriangularSet := proc(S::list, vars::list)
   local lvars:
   # An empty set is natural to be a triangluar set
   if nops(S) = 0 then
      return true:
   end if:
   # A set cannot be a triangular set if constant term exists
   if nops(select(xx -> type(xx, constant), S)) = 0 then
      lvars := MakeUnique(map(xx -> MainVariable(xx, PolynomialRing(vars)), S)):
      if nops(lvars) = nops(S) then
         if Sorted(map(xx -> Search(xx, vars), lvars), `<`) then
            return true:
         end if:
      end if:
   end if:
   return false:
end proc:

# Calculate the chain remainder of a polynomial over a triangluar set
CPrem := proc(f::polynom, T::list, vars::list)
   local PR, r, lvar, i:
   PR := PolynomialRing(vars):
   r := f:
   for i to nops(T) do
      lvar := MainVariable(T[i], PR):
      r := expand(prem(r, T[i], lvar)):
   end do:
   return r:
end proc:

# Judge whether a triangular set is reduced
isReduced := proc(T::list, vars::list)
   local r, i:
   # A set is reduced if it is empty or has only one polynomial
   if nops(T) > 1 then
      for i to nops(T) - 1 do
         r := CPrem(T[i], T[i + 1..-1], vars[i + 1..-1]):
         if r - T[i] <> 0 then
            return false:
         end if:
      end do:
   end if:
   return true:
end proc:

# Judge whether a set is an ascending set
isAscendingSet := proc(S::list, vars::list)
   if isTriangularSet(S, vars) then
      return isReduced(S, vars):
   end if:
   return false:
end proc:

# evalb(isCharSet([x+y^2-1, x^2+y-1], [y^2+x-1, y^4-2*y^2+y]) = true);
# evalb(isCharSet([x+y^2-1, x+y-1], [y^2+x-1, y*(y^3-2*y+1)]) = false);
# evalb(isCharSet([-x0^5*x1^5-6*x0^3*x1^7, 5*x0^7*x1^5+7*x1^8], [1080*x0*x1^63-7*x1^60, 6998400*x1^170+49*x1^162]) = true);
# evalb(isCharSet([-9*x0^6*x1^3+7*x1^6, 2*x0^8*x1^2-8*x0^4*x1^6], [-144*x1^24+7*x0^2*x1^19, 5184*x1^57-49*x1^51]) = true);
isCharSet := proc(sys::list, predict::list)
   local vars, r, gb, nf:
   vars := sort([op(indets(sys))]):
   # Any extra variable is illegal
   if `subset`(indets(predict), {op(vars)}) then
      # Empty characteristic set
      if nops(predict) = 0 then
         return not IsProper(PolynomialIdeal(sys)):
      # A characteristic set must be an ascending set
      elif isAscendingSet(predict, vars) then
         r := MakeUnique(map(xx -> CPrem(xx, predict, vars), sys)):
         # Zero remainder in pseudodivison of any polynomial in the system over the set
         if nops(r) = 1 and r[1] = 0 then
            gb := Basis(sys, tdeg(op(vars))):   # Gr\"obner basis
            nf := MakeUnique(map(xx -> NormalForm(xx, gb, tdeg(op(vars))), predict)):
            # The set represents solution set of the system
            if nops(nf) = 1 and nf[1] = 0 then
               return true:
            end if:
         end if:
      end if:
   end if:
   return false:
end proc: