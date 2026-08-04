import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class Estado:
    
    piso: int          
    demanda: int       


@dataclass
class MDP:
    pisos: tuple = (1, 2, 3, 4, 5)
    demandas: tuple = (0, 1)
    acciones: tuple = ("subir", "bajar", "permanecer")
    gamma: float = 0.95
    theta: float = 1e-6

    estados: list = field(default_factory=list)
    tabla_transiciones: dict = field(default_factory=dict)
    tabla_recompensas: dict = field(default_factory=dict)

    def __post_init__(self):
        self.estados = [Estado(p, d) for p in self.pisos for d in self.demandas]
        for s in self.estados:
            for a in self.acciones:
                s_siguiente, r = self._calcular_transicion(s, a)
                self.tabla_transiciones[(s, a)] = s_siguiente
                self.tabla_recompensas[(s, a)] = r

    def _calcular_transicion(self, estado: Estado, accion: str):
        if accion == "subir":
            nuevo_piso = min(estado.piso + 1, 5)
        elif accion == "bajar":
            nuevo_piso = max(estado.piso - 1, 1)
        else:  
            nuevo_piso = estado.piso

        es_movimiento = accion in ("subir", "bajar")

        if estado.demanda == 1:
            if es_movimiento:
                nueva_demanda, recompensa = 0, 10.0     
            else:
                nueva_demanda, recompensa = 1, -1.0     
        else:  
            if es_movimiento:
                nueva_demanda, recompensa = 0, -0.5     
            else:
                nueva_demanda, recompensa = 0, 0.0      

        return Estado(nuevo_piso, nueva_demanda), recompensa


@dataclass
class Resultado:
    nombre: str
    V: dict
    politica: dict
    iteraciones: int
    tiempo: float
    aplicaciones_bellman: Optional[int] = None

    def imprimir(self):
        print(f"  {self.nombre}")
        print(f"Iteraciones hasta convergencia   : {self.iteraciones}")
        if self.aplicaciones_bellman is not None:
            print(f"Aplicaciones totales de Bellman  : {self.aplicaciones_bellman}")
        print(f"Tiempo de ejecución : {self.tiempo:.6f}")
        print()
        print(f"{'Estado (piso, demanda)':<25}{'V*(s)':>12}   π*(s)")
        print("-" * 62)
        for s, v in self.V.items():
            etiqueta = f"({s.piso}, {s.demanda})"
            print(f"{etiqueta:<25}{v:>12.6f}   {self.politica[s]}")


def evaluacion_de_politica(mdp: MDP, politica: dict, V_inicial: dict):

    V = dict(V_inicial)
    aplicaciones_bellman = 0

    while True:
        delta = 0.0
        for s in mdp.estados:
            v_previo = V[s]
            a = politica[s]
            s_siguiente = mdp.tabla_transiciones[(s, a)]
            r = mdp.tabla_recompensas[(s, a)]
            V[s] = r + mdp.gamma * V[s_siguiente]     
            aplicaciones_bellman += 1
            delta = max(delta, abs(v_previo - V[s]))
        if delta < mdp.theta:
            break

    return V, aplicaciones_bellman

def iteracion_de_politica(mdp: MDP) -> Resultado:
    politica = {s: "permanecer" for s in mdp.estados}
    V = {s: 0.0 for s in mdp.estados}

    iteraciones_externas = 0
    total_aplicaciones_bellman = 0

    inicio = time.perf_counter()
    while True:
        iteraciones_externas += 1

        V, apps_eval = evaluacion_de_politica(mdp, politica, V)
        total_aplicaciones_bellman += apps_eval

        politica_estable = True
        for s in mdp.estados:
            accion_previa = politica[s]

            valores_q = {}
            for a in mdp.acciones:
                s_siguiente = mdp.tabla_transiciones[(s, a)]
                r = mdp.tabla_recompensas[(s, a)]
                valores_q[a] = r + mdp.gamma * V[s_siguiente]
                total_aplicaciones_bellman += 1

            mejor_accion = max(valores_q, key=valores_q.get)
            politica[s] = mejor_accion
            if mejor_accion != accion_previa:
                politica_estable = False

        if politica_estable:
            break
    tiempo = time.perf_counter() - inicio

    return Resultado(
        nombre="POLICY ITERATION",
        V=V,
        politica=politica,
        iteraciones=iteraciones_externas,
        tiempo=tiempo,
        aplicaciones_bellman=total_aplicaciones_bellman,
    )


def iteracion_de_valor(mdp: MDP) -> Resultado:
    V = {s: 0.0 for s in mdp.estados}
    iteraciones = 0

    inicio = time.perf_counter()
    while True:
        iteraciones += 1
        delta = 0.0
        V_nuevo = {}

        for s in mdp.estados:
            valores_q = []
            for a in mdp.acciones:
                s_siguiente = mdp.tabla_transiciones[(s, a)]
                r = mdp.tabla_recompensas[(s, a)]
                valores_q.append(r + mdp.gamma * V[s_siguiente])
            V_nuevo[s] = max(valores_q)
            delta = max(delta, abs(V[s] - V_nuevo[s]))

        V = V_nuevo
        if delta < mdp.theta:
            break

    politica_optima = {}
    for s in mdp.estados:
        valores_q = {}
        for a in mdp.acciones:
            s_siguiente = mdp.tabla_transiciones[(s, a)]
            r = mdp.tabla_recompensas[(s, a)]
            valores_q[a] = r + mdp.gamma * V[s_siguiente]
        politica_optima[s] = max(valores_q, key=valores_q.get)
    tiempo = time.perf_counter() - inicio

    return Resultado(
        nombre="VALUE ITERATION",
        V=V,
        politica=politica_optima,
        iteraciones=iteraciones,
        tiempo=tiempo,
    )


if __name__ == "__main__":
    mdp = MDP()

    resultado_pi = iteracion_de_politica(mdp)
    resultado_vi = iteracion_de_valor(mdp)

    resultado_pi.imprimir()
    resultado_vi.imprimir()