import matplotlib.pyplot as plt
import numpy as np
import pandas


def print_results_pretty(results_dict):
    """
    Nimmt das Ergebnis-Dictionary von Rohrreaktor und druckt es in einem leserlichen,
    strukturierten Format aus.
    """
    print("\n--- AUSWERTUNG DER REAKTORENDATEN ---")

    # Abschnitt 1: Die einzelnen Kennzahlen
    print("\nKENNGRÖSSEN DES REAKTORS:")
    print("-----------------------------------------")
    for key, value in results_dict.items():
        if not isinstance(value, list):  # Nur Werte drucken, die keine Listen sind
            # Formatierung für eine saubere, linksbündige Ausrichtung der Bezeichner
            print(f"{key:<38}: {value:.5f}")

    # Abschnitt 2: Die zeitabhängigen Datenlisten
    print("\n\nZEITABHÄNGIGE DATEN (MESSKURVE):")
    print("-----------------------------------------")

    t_list = results_dict.get("Zeitachse t [min]", [])
    c_t_list = results_dict.get("Konzentrationskurve c(t) [mol/L]", [])

    # Spaltenüberschriften für die Tabelle
    print(f"{'Zeit [min]':<15} | {'c(t) [mol/L]':<20}")
    print("-----------------------------------------")

    # Daten Zeile für Zeile als Tabelle ausgeben
    if t_list and c_t_list:
        for t_val, c_val in zip(t_list, c_t_list):
            print(f"{t_val:<15.2f} | {c_val:<20.6f}")
    else:
        print("Keine Listendaten zum Anzeigen gefunden.")

    print("\n--- ENDE DER AUSWERTUNG ---")


## numerische integration
def nIntegration(datax, datay):
    """
    Berechnet das numerische Integral von diskreten Datenpunkten mittels der Trapezregel.

    Diese Funktion approximiert die Fläche unter der Kurve, die durch die
    (x, y)-Punkte definiert wird. Sie zerlegt die Fläche in eine Reihe von
    Trapezen (für jedes Intervall zwischen zwei x-Punkten) und summiert
    deren einzelne Flächeninhalte, um das Gesamtintegral zu erhalten.

    Args:
        datax (list): Eine Liste mit den x-Koordinaten der Datenpunkte.
                      Die Werte sollten sortiert sein (entweder auf- oder absteigend).
                      Muss die gleiche Länge wie datay haben.
        datay (list): Eine Liste mit den y-Koordinaten der Datenpunkte.
                      Muss die gleiche Länge wie datax haben.

    Returns:
        float: Der approximierte Wert des Integrals, also die Gesamtfläche unter
               der durch die Punkte definierten Kurve.

    Raises:
        ValueError: Wenn datax und datay nicht die gleiche Länge haben,
                    sodass keine (x,y)-Paare gebildet werden können.
    """
    if len(datax) != len(datay):
        raise ValueError(
            "Die Eingabelisten datax und datay müssen die gleiche Länge haben."
        )
    if len(datax) < 2:
        return 0.0
    xdif = [j - i for i, j in zip(datax, datax[1:])]
    ydif = [(i + j) / 2 for i, j in zip(datay, datay[1:])]
    inte = [i * j for i, j in zip(xdif, ydif)]
    return sum(inte)


def ndifferential(datax, datay):

    xdiff = [j - i for i, j in zip(datax, datax[1:])]
    ydiff = [j - i for i, j in zip(datay, datay[1:])]

    return xdiff, ydiff


def plotshow(
    datax,
    datay,
    title=None,
    label=None,
    grid=True,
    xlabel=None,
    ylabel=None,
    filepath=None,
    show=True,
    marker=None,
    linestyle="-",
):
    """
    Erstellt und zeigt einen flexiblen Plot mit optionalen Konfigurationen.

    Args:
        datax (list): Eine Liste mit den Daten für die x-Achse.
        datay (list): Eine Liste mit den Daten für die y-Achse.
        title (str, optional): Überschrift für das Diagramm.
        label (str, optional): Die Beschriftung für die Datenreihe. Standard ist None.
        grid (bool, optional): Ob ein Gitter angezeigt werden soll. Standard ist True.
        xlabel (str, optional): Die Beschriftung für die x-Achse. Standard ist None.
        ylabel (str, optional): Die Beschriftung für die y-Achse. Standard ist None.
        filepath (str, optional): Der Dateipfad zum Speichern (nur der Ordner).
        show (bool, optional): Ob der Plot direkt angezeigt werden soll. Standard ist True.
        marker (str, optional): Der Stil des Markers (z.B. 'o', 'x', 's').
                                Standard ist None (kein Marker).
        linestyle (str, optional): Der Stil der Linie (z.B. '-', '--', ':').
                                   'None' für keine Linie. Standard ist '-' (durchgezogene Linie).
    """

    plt.figure(figsize=(10, 6))
    plt.grid(grid)

    # 1. Verbessert: X- und Y-Achsen werden in der üblichen Reihenfolge geplottet.
    # 2. Verbessert: Die neuen Parameter marker und linestyle werden übergeben.
    # 3. Korrigiert: Das 'label' wird korrekt als Keyword-Argument übergeben.
    plt.plot(datax, datay, label=label, marker=marker, linestyle=linestyle)

    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)

    # Eine Legende wird nur angezeigt, wenn ein Label vorhanden ist.
    if label:
        plt.legend()

    if filepath and title:
        # Erstellt einen sicheren Dateinamen aus dem Titel
        sicherer_titel = "".join(
            c for c in title if c.isalnum() or c in (" ", "_")
        ).rstrip()
        plt.savefig(f"{filepath}/{sicherer_titel}.png")

    if show:
        plt.show()


def oldplotshow(
    datax,
    datay,
    title=None,
    label=None,
    grid=True,
    xlabel=None,
    ylabel=None,
    filepath=None,
    show=True,
):
    """
    Erstellt und zeigt einen einfachen Plot mit optionalen Konfigurationen.

    Args:
        datax (list): Eine Liste mit den Daten für die x-Achse.
        datay (list): Eine Liste mit den Daten für die y-Achse.
        title (str, optional): Überschrift für das Diagramm.
        label (str, optional): Die Beschriftung für die Datenreihe. Standard ist None.
        grid (bool, optional): Ob ein Gitter angezeigt werden soll. Standard ist True.
        xlabel (str, optional): Die Beschriftung für die x-Achse. Standard ist None.
        ylabel (str, optional): Die Beschriftung für die y-Achse. Standard ist None.
        save (bool, optional): Ob der Plot als Datei gespeichert werden soll. Standard ist False.
        filepath (str, optional): Der Dateipfad zum Speichern. Erforderlich, wenn save=True.
                                 Standard ist None.
    """

    plt.figure(figsize=(10, 6))
    plt.grid(grid)
    plt.title(title)
    if label:
        plt.plot(datay, datax, label)
        plt.legend()
    else:
        plt.plot(datay, datax)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)

    if filepath:
        plt.savefig(filepath + f"/{title}")
    if show == True:
        plt.show()


## braucht noch arbeit id say
def tocsv(data, title="Untitled.csv"):

    pandas.DataFrame(data).to_csv(title)


def datamake(filepath, sep=";", skiprows=None, header=None):
    """
    Liest eine CSV-Datei ein und konvertiert jede Spalte in eine Liste mit numerischen oder String-Werten.

    Diese Funktion verarbeitet tabellarische Daten aus CSV-Dateien und konvertiert
    jede Spalte separat. Für jede Zelle wird versucht, den Wert in einen Float zu konvertieren.
    Falls dies fehlschlägt (z.B. bei Textwerten oder speziellen Zeichen), wird der
    Originalwert als String beibehalten. Das Ergebnis ist eine Liste von Spaltenlisten.

    Args:
        filepath (str): Pfad zur CSV-Datei.
        sep (str, optional): Trennzeichen für die CSV-Datei. Standardwert: ";".
        skiprows (int/list, optional): Zeilen, die am Anfang übersprungen werden sollen.
                                       Beispiel: skiprows=2 überspringt die ersten 2 Zeilen.
        header (int, optional): Zeilennummer (0-indiziert), die als Spaltenüberschriften dient.
                                None bedeutet keine Überschriften. Beispiel: header=0.

    Returns:
        list: Liste von Spaltenlisten. Jede Spaltenliste enthält konvertierte Werte:
              - Float-Werte bei erfolgreicher Konvertierung
              - Strings bei nicht-konvertierbaren Werten
              Beispielstruktur: [[col1_val1, col1_val2], [col2_val1, col2_val2]]

    Raises:
        FileNotFoundError: Wenn die angegebene Datei nicht existiert.
        pandas.errors.ParserError: Bei Problemen mit dem CSV-Parsing.
        ValueError: Bei inkonsistenten Datenformaten.

    Beispiel:
        >>> datamake("data.csv", sep=",", skiprows=1, header=0)
        [
            [1.0, 2.0, 3.0],                   # Konvertierte Float-Werte
            ["Header", "Value", "N/A"],          # Beibehaltene Strings
            [0.5, 1.5, 2.5]
        ]
    """
    data = pandas.read_csv(filepath, sep=sep, skiprows=skiprows, header=header)
    dlist = []
    dlist = []
    for index, col in enumerate(
        data.columns
    ):  # 2. Iteration über Spaltennamen statt Indizes
        s = data[col]
        dlist.append([])
        for i in s:
            # 3. Versuch, die Spalte in Float zu konvertieren
            try:
                converted = float(i)
                dlist[index].append(converted)
            except ValueError:
                converted = str(i)
                dlist[index].append(converted)

    return dlist


def sigmoidalmake(n):
    """
    Normalisiert eine numerische Liste durch Division jedes Elements durch den letzten Wert.

    Diese Funktion transformiert eine Liste von Zahlen, indem jedes Element durch den
    letzten Wert der Liste geteilt wird. Dies erzeugt eine normalisierte Darstellung
    der Daten, bei der alle Werte auf den Endwert skaliert sind.

    Typische Anwendungen:
    - Vorbereitung von Daten für sigmoide Transformationen
    - Skalierung von Zeitreihen auf den Endwert
    - Erstellung relativer Vergleichsmetriken

    Args:
        n (list): Eine Liste numerischer Werte (int oder float).
                  Muss mindestens ein Element enthalten.

    Returns:
        list: Normalisierte Liste, wo jedes Element = Originalwert / letzter Wert.
              Der letzte Wert der Ergebnisliste ist immer 1.0.

    Raises:
        ZeroDivisionError: Wenn das letzte Element der Liste 0 ist.
        IndexError: Wenn die Eingabeliste leer ist.
        TypeError: Wenn nicht-numerische Werte enthalten sind.

    Beispiel:
        >>> sigmoidalmake([2, 4, 8, 16])
        [0.125, 0.25, 0.5, 1.0]  # 2/16=0.125, 4/16=0.25, etc.

        >>> sigmoidalmake([10, 20, 30])
        [0.333, 0.666, 1.0]  # 10/30≈0.333, 20/30≈0.666

    Hinweis:
        Die Funktion ist nicht symmetrisch zum Ursprung und eignet sich speziell für
        Daten, die auf einen Endwert zustreben (z.B. Sättigungskurven).
    """
    konvdata = [i / n[-1] for i in n]
    return konvdata


def minussigmoidalmake(n):
    """
    Berechnet die komplementäre Normalisierung einer numerischen Liste relativ zum Endwert.

    Diese Funktion erzeugt eine invertierte Version der sigmoidalen Normalisierung,
    indem jedes Element durch den letzten Wert geteilt und dann von 1 subtrahiert wird.
    Das Ergebnis zeigt den relativen Abstand jedes Elements zum Endwert an.

    Mathematische Formel:
        result[i] = 1 - (n[i] / n[-1])

    Typische Anwendungen:
    - Analyse von Annäherungsprozessen an einen Grenzwert
    - Transformation von Wachstumsdaten in "Restabstand"-Darstellung
    - Vorbereitung für bestimmte statistische Analysen

    Args:
        n (list): Eine Liste numerischer Werte (int oder float).
                  Muss mindestens ein Element enthalten.

    Returns:
        list: Transformierte Liste mit Werten zwischen (-∞, 1].
              Der letzte Wert ist immer 0 (da 1 - n[-1]/n[-1] = 0).

    Raises:
        ZeroDivisionError: Wenn das letzte Element 0 ist.
        IndexError: Bei leerer Eingabeliste.
        TypeError: Bei nicht-numerischen Werten.

    Beispiele:
        >>> minussigmoidalmake([10, 20, 30, 40])
        [0.75, 0.5, 0.25, 0.0]  # 1-(10/40)=0.75, 1-(20/40)=0.5, etc.

        >>> minussigmoidalmake([5, 5, 5])
        [0.0, 0.0, 0.0]  # Alle Werte gleich dem Endwert → Ergebnis 0

    Besonderheiten:
        - Kann negative Werte produzieren, wenn Elemente > Endwert sind
        - Sensitiv für kleine Endwerte (Division durch kleine Zahlen)
    """
    konvdata = [1 - (i / n[-1]) for i in n]
    return konvdata


def verweilzeitUmsatzmake(tau, k0, Ea, T, C_0i, ii):
    """
    Berechnet den Umsatzgrad einer chemischen Reaktion in einem kontinuierlichen Rührkesselreaktor (CSTR)
    unter Berücksichtigung der Verweilzeitverteilung und Arrhenius-Kinetik.

    Die Funktion modelliert den Fortschritt einer Reaktion n-ter Ordnung mit temperaturabhängiger
    Geschwindigkeitskonstante und berechnet iterativ den Umsatzgrad für jede Verweilzeit.

    Theoretischer Hintergrund:
    - Arrhenius-Gleichung: k = k0 * exp(-Ea/(R*T))
    - CSTR-Umsatzgradgleichung: F = (1 + 2a - sqrt(1 + 4a(1 - F_prev))) / (2a)
    - Mit a = τ * k * C0

    Args:
        tau (list): Liste der Verweilzeiten [s]
        k0 (float): Präexponentieller Faktor der Arrhenius-Gleichung [1/s]
        Ea (float): Aktivierungsenergie [J/mol]
        T (float): Temperatur [K]
        _Oi (list): Startkonzentrationen der Komponenten [mol/m³]
        ii (int): Anzahl der Iterationen

    Returns:
        list: Umsatzgrade F für jede Verweilzeit (dimensionslos zwischen 0 und 1)

    Raises:
        ValueError: Bei negativen Temperaturen oder Konzentrationen
        TypeError: Bei nicht-numerischen Eingaben
        ZeroDivisionError: Wenn a = 0 berechnet wird

    Beispiel:
        >>> verweilzeitUmsatzmakel(
                tau=[10, 20, 30],
                k0=1e8,
                Ea=75000,
                T=350,
                _Oi=[1.0, 0.8],
                ii=3
            )
        [0.0, 0.382, 0.541]  # Umsatzgrade bei verschiedenen Verweilzeiten

    Wichtige Variablen:
        R_0 = 8.314  # Universelle Gaskonstante [J/(mol*K)]
        C_0i = _Oi    # Initialkonzentrationen [mol/m³]
        F_i  = [0]    # Umsatzgrad-Array initialisiert mit F(t=0)=0
    """
    R_0 = 8.314
    k = k0 * (np.e) ** (-Ea / (R_0 * T))
    tauexp = tau
    F_i = [0]
    for i in range(ii):
        a = tauexp[i] * k * C_0i[i] * 60
        F = (1 + 2 * a - np.sqrt(1 + 4 * a * (1 - F_i[i]))) / (2 * a)
        F_i.append(F)
        C_0i.append(F * C_0i[i])
    return F_i


def molenmake(CA_0, CA_inf, W_0, W_inf, conductivity):
    """
    Berechnet die Konzentration einer Komponente zu verschiedenen Zeitpunkten anhand von Leitfähigkeitsmessungen.

    Diese Funktion verwendet die gemessene Leitfähigkeit, um die Konzentration einer Komponente
    (z.B. eines Reaktanten oder Produkts) zu berechnen. Die Umrechnung basiert auf einer linearen
    Interpolation zwischen Anfangs- und Endwerten der Leitfähigkeit und Konzentration.

    Args:
        CA_0 (float): Anfangskonzentration der Komponente.
        CA_inf (float): Endkonzentration der Komponente.
        W_0 (float): Anfangswert der Leitfähigkeit.
        W_inf (float): Endwert der Leitfähigkeit.
        conductivity (list): Liste der gemessenen Leitfähigkeitswerte.

    Returns:
        list: Liste der berechneten Konzentrationen zu jedem Leitfähigkeitswert.
    """
    CA_t = [
        (W - W_inf) / (W_0 - W_inf) * (CA_0 - CA_inf) + CA_inf for W in conductivity
    ]

    return CA_t


def Leitfähigkeitsumsatzmake(CA_0, CA_inf, W_0, W_inf, conductivity):

    CA_t = [
        (W - W_inf) / (W_0 - W_inf) * (CA_0 - CA_inf) + CA_inf for W in conductivity
    ]


def Manuelleauswertung(E_t, t, mol_in, Volstrom, c_in, k0, Ea, T):
    """
    Führt die vollständige Auswertung der Verweilzeitdaten eines Rohreaktors durch.

    Args:
        E_t (list): Liste der E(t)-Werte (normierte Verweilzeitdichte).
        t (list): Liste der Zeitpunkte t.
        mol_in (float): Eingegebene Stoffmenge des Tracers (n₀).
        Volstrom (float): Volumenstrom durch den Reaktor (V̇).

    Returns:
        dict: Ein Dictionary mit allen berechneten Kenngrößen.
        "Mittlere Verweilzeit t_bar [min]": mü1,
        "Zweites Moment M2 [min^2]": mü2,
        "Varianz sigma^2 [min^2]": sigsq,
        "Standardabweichung sigma [min]": sig,
        "Dimensionslose Varianz sigma_theta^2": sigma_theta_sq,
        "Dispersionszahl D/(uL)": D_uL,
        "Effektives Reaktorvolumen V_eff [Einheit von Volstrom * s]": V_eff,
        "Zeitachse t [min]": t,
        "Konzentrationskurve c(t)": c_t,
        "theoretischer Umsatz": f_i
    """

    # Annahme: nIntegration ist die Trapezregel, hier mit numpy implementiert
    E_tn = E_t / nIntegration(t, E_t)
    # Berechnung der Momente und Kennzahlen
    mü1 = nIntegration(t, [a * b for a, b in zip(E_tn, t)])
    mü2 = nIntegration(t, [a * (b**2) for a, b in zip(E_tn, t)])
    sigsq = mü2 - mü1**2
    sig = sigsq**0.5
    sigma_theta_sq = sigsq / (mü1**2)
    D_uL = (-2 + (4 + 32 * sigma_theta_sq) ** 0.5) / 16
    V_eff = mü1 * Volstrom
    c_t = [a * mol_in / Volstrom for a in E_tn]
    R_0 = 8.314
    k = k0 * (np.e) ** (-Ea / (R_0 * T))
    f_i = (c_in * mü1 * 60 * k) / (1 + (c_in * mü1 * 60 * k))
    # Ergebnisse in einem Dictionary sammeln
    ergebnisse = {
        "Mittlere Verweilzeit t_bar [min]": mü1,
        "Zweites Moment M2 [min^2]": mü2,
        "Varianz sigma^2 [min^2]": sigsq,
        "Standardabweichung sigma [min]": sig,
        "Dimensionslose Varianz sigma_theta^2": sigma_theta_sq,
        "Dispersionszahl D/(uL)": D_uL,
        "Effektives Reaktorvolumen V_eff [Einheit von Volstrom * s]": V_eff,
        "Zeitachse t [min]": t,
        "theoretischer Umsatz": f_i,
        "Konzentrationskurve c(t)": list(c_t),
    }

    print_results_pretty(ergebnisse)


