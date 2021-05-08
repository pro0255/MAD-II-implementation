from view.Application import Application
from logic.Controller import Controller

"""    
    1. -> Predikce linků – Adamic-Adar, Resource Allocation Index, Cosine similarity, Sorensen
            Index, CAR-based Common Neighbor Index. Posouzení výkonnosti metody pro různé
            prahy.

    2. -> Implementace bude obsahovat jednoduché uživatelské rozhraní (formulář) pro výběr souboru s daty, pro nastavení parametrů          implementované metody a pro výstup popisující základní vlastnosti sítě (počet hran a vrcholů, průměrný a maximální stupeň, průměrný shlukovací koeficient, počet komponent s alespoň dvěma vrcholy, počet izolovaných vrcholů, velikost největší komponenty), čas výpočtu a výsledky získané aplikací metody tedy:


        c) Predikce linků – confusion matice a míry na ní založené (sensitivita, specificita, precisssion, fallout, accuracy).
"""

c = Controller()
app = Application(c)
app.start()
