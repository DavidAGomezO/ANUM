-- Función para pasar de binario a base 10

binInt :: Int -> Int
binInt s = go 0 0 s
  where
    go n m k
      | k == 0    = n
      | otherwise = go (n + 2^m * (k `mod` 10)) (m+1) (k `div` 10)

binFloat :: Floating a => a -> a
binFloat s = go 0 1 s
  where
    go n m k
      | k == 0    = n
      | otherwise = go (n + 2**(-m)*(floor 10*k)) (m+1) (k - (floor k))
main :: IO ()
main = do
  print( binFloat 0.1 )