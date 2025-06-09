#!/bin/bash

# Fare konumları
declare -a konumlar=("100 200" "300 400" "500 600")

# Sonsuz döngü
while true; do
    for konum in "${konumlar[@]}"; do
        # Fareyi konuma götür
        xdotool mousemove $konum
        
        # Tıklama yap
        xdotool click 1  # 1: sol tıklama
        
        # 1 saniye bekle
        sleep 10
    done
done
