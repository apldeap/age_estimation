function mse(outputs, targets)
   local mseError = torch.add(outputs, -1 * targets)
   mseError:pow(2)
   mseError = torch.sum(mseError)
   mseError = mseError / outputs:size(1)
   return mseError
end