-- Función para pasar de binario a base 10

binDec :: Int -> Int
binDec s = go 0 0 s
  where
    go n m k
      | k == 0    = n
      | otherwise = go (n + 2^m * (k `mod` 10)) (m+1) (k `div` 10)

main :: IO ()
main = do
  print( binDec 101101 )