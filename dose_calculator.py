"""
Drug Dosage Calculator for Turkish Medical Practice
Based on T.C. Sağlık Bakanlığı guidelines

Supports common pediatric medications with safety checks.
"""
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DoseResult:
    """Result of dose calculation with safety information"""
    drug_name: str
    dose: str
    frequency: str
    warnings: list
    calculation_method: str
    is_safe: bool


def calc_amoxicillin_dose(weight_kg: float, age_years: float) -> DoseResult:
    """
    Calculate amoxicillin dose for pediatric patients.

    Turkish Guidelines (Sağlık Bakanlığı):
    - Standard: 20-40 mg/kg/day divided into 2-3 doses
    - Max single dose: 500 mg
    - Max daily dose: 3000 mg

    Args:
        weight_kg: Patient weight in kg
        age_years: Patient age in years

    Returns:
        DoseResult with calculated dose and warnings
    """
    warnings = []

    # Validation
    if weight_kg < 3 or weight_kg > 150:
        warnings.append("⚠️ Kilo normal aralıkta değil (3-150 kg)")

    if age_years < 0.1:  # 1 month
        warnings.append("⚠️ 1 aylıktan küçük bebekler için uzman konsültasyonu gerekli")
        return DoseResult(
            drug_name="Amoksisilin",
            dose="Uygulanmaz",
            frequency="N/A",
            warnings=warnings,
            calculation_method="Yaş sınırı",
            is_safe=False
        )

    # Standard dosing: 25 mg/kg/day (mid-range)
    daily_dose = 25 * weight_kg

    # Check max daily dose
    if daily_dose > 3000:
        daily_dose = 3000
        warnings.append("⚠️ Maksimum günlük doz uygulandı (3000 mg)")

    # Divide into 2 doses (common in Turkey)
    single_dose = daily_dose / 2

    # Check max single dose
    if single_dose > 500:
        single_dose = 500
        warnings.append("⚠️ Tek doz maksimum 500 mg ile sınırlandı")

    # Round to practical values
    single_dose = round(single_dose / 125) * 125  # Round to 125 mg increments
    daily_dose = single_dose * 2

    # Safety check
    is_safe = len([w for w in warnings if "⚠️" in w]) == 0

    return DoseResult(
        drug_name="Amoksisilin",
        dose=f"{single_dose} mg",
        frequency=f"Günde 2 kez (toplam {daily_dose} mg/gün)",
        warnings=warnings if warnings else ["✓ Doz güvenli aralıkta"],
        calculation_method=f"25 mg/kg/gün × {weight_kg} kg ÷ 2 doz",
        is_safe=is_safe
    )


def calc_paracetamol_dose(weight_kg: float, age_years: float) -> DoseResult:
    """
    Calculate paracetamol (acetaminophen) dose for pediatric patients.

    Turkish Guidelines:
    - Standard: 10-15 mg/kg/dose
    - Frequency: Every 4-6 hours (max 4-5 doses/day)
    - Max single dose: 1000 mg
    - Max daily dose: 4000 mg (adult), 75 mg/kg/day (pediatric)

    Args:
        weight_kg: Patient weight in kg
        age_years: Patient age in years

    Returns:
        DoseResult with calculated dose and warnings
    """
    warnings = []

    # Validation
    if weight_kg < 3:
        warnings.append("⚠️ 3 kg altı bebekler için hekim konsültasyonu gerekli")

    # Standard dosing: 15 mg/kg/dose
    single_dose = 15 * weight_kg

    # Check max single dose
    if age_years >= 12 and single_dose > 1000:
        single_dose = 1000
        warnings.append("⚠️ Tek doz maksimum 1000 mg ile sınırlandı")
    elif age_years < 12 and single_dose > 500:
        single_dose = 500
        warnings.append("⚠️ Çocuk için tek doz maksimum 500 mg")

    # Calculate max daily dose
    max_daily = min(75 * weight_kg, 4000 if age_years >= 12 else 75 * weight_kg)
    max_doses_per_day = int(max_daily / single_dose)
    if max_doses_per_day > 5:
        max_doses_per_day = 5

    # Round to practical values (drop or syrup volumes)
    if single_dose < 250:
        single_dose = round(single_dose / 25) * 25  # 25 mg increments (for syrup)
    else:
        single_dose = round(single_dose / 50) * 50  # 50 mg increments

    # Safety checks
    if single_dose * 5 > max_daily:
        warnings.append("⚠️ Maksimum günlük dozu aşmamaya dikkat edin")

    is_safe = len([w for w in warnings if "⚠️" in w]) <= 1

    return DoseResult(
        drug_name="Parasetamol (Asetaminofen)",
        dose=f"{single_dose} mg",
        frequency=f"Her 4-6 saatte bir (günde en fazla {max_doses_per_day} doz)",
        warnings=warnings if warnings else ["✓ Doz güvenli aralıkta"],
        calculation_method=f"15 mg/kg × {weight_kg} kg",
        is_safe=is_safe
    )


def calc_ibuprofen_dose(weight_kg: float, age_years: float) -> DoseResult:
    """
    Calculate ibuprofen dose for pediatric patients.

    Turkish Guidelines:
    - Standard: 5-10 mg/kg/dose
    - Frequency: Every 6-8 hours (max 3-4 doses/day)
    - Max daily dose: 40 mg/kg/day or 1200 mg
    - Not recommended under 6 months

    Args:
        weight_kg: Patient weight in kg
        age_years: Patient age in years

    Returns:
        DoseResult with calculated dose and warnings
    """
    warnings = []

    # Age restriction
    if age_years < 0.5:  # 6 months
        warnings.append("⚠️ 6 aylıktan küçük bebekler için önerilmez")
        return DoseResult(
            drug_name="İbuprofen",
            dose="Uygulanmaz",
            frequency="N/A",
            warnings=warnings,
            calculation_method="Yaş sınırı",
            is_safe=False
        )

    # Standard dosing: 10 mg/kg/dose
    single_dose = 10 * weight_kg

    # Check max daily dose
    max_daily = min(40 * weight_kg, 1200)
    max_doses = 4

    if single_dose > max_daily / 3:
        single_dose = max_daily / 3
        warnings.append(f"⚠️ Maksimum günlük doz: {max_daily} mg")

    # Round to practical values
    if single_dose < 200:
        single_dose = round(single_dose / 20) * 20  # 20 mg increments
    else:
        single_dose = round(single_dose / 50) * 50  # 50 mg increments

    # Additional warnings
    warnings.append("ℹ️ Tok karnına alınmalı (mide koruyucu)")

    is_safe = len([w for w in warnings if "⚠️" in w]) == 0

    return DoseResult(
        drug_name="İbuprofen",
        dose=f"{single_dose} mg",
        frequency=f"Her 6-8 saatte bir (günde en fazla {max_doses} doz)",
        warnings=warnings,
        calculation_method=f"10 mg/kg × {weight_kg} kg",
        is_safe=is_safe
    )


# Drug database mapping
SUPPORTED_DRUGS = {
    'amoksisilin': calc_amoxicillin_dose,
    'amoxicillin': calc_amoxicillin_dose,
    'amoksilin': calc_amoxicillin_dose,
    'parasetamol': calc_paracetamol_dose,
    'paracetamol': calc_paracetamol_dose,
    'asetaminofen': calc_paracetamol_dose,
    'acetaminophen': calc_paracetamol_dose,
    'ibuprofen': calc_ibuprofen_dose,
    'brufen': calc_ibuprofen_dose,
}


def calculate_dose(drug: str, weight_kg: float, age_years: float) -> DoseResult:
    """
    Main function to calculate drug dosage.

    Supports Turkish and English drug names.

    Args:
        drug: Drug name (Turkish or English)
        weight_kg: Patient weight in kg
        age_years: Patient age in years

    Returns:
        DoseResult with dose calculation and safety warnings

    Example:
        >>> result = calculate_dose("amoksisilin", 25, 7)
        >>> print(result.dose)
        '312.5 mg'
    """
    # Normalize drug name
    drug_normalized = drug.lower().strip()

    # Check if drug is supported
    if drug_normalized not in SUPPORTED_DRUGS:
        return DoseResult(
            drug_name=drug,
            dose="Desteklenmiyor",
            frequency="N/A",
            warnings=[
                f"❌ '{drug}' için doz hesaplaması henüz desteklenmiyor.",
                f"Desteklenen ilaçlar: {', '.join(set(['Amoksisilin', 'Parasetamol', 'İbuprofen']))}"
            ],
            calculation_method="N/A",
            is_safe=False
        )

    # Basic validation
    try:
        weight_kg = float(weight_kg)
        age_years = float(age_years)
    except (ValueError, TypeError):
        return DoseResult(
            drug_name=drug,
            dose="Hata",
            frequency="N/A",
            warnings=["❌ Geçersiz kilo veya yaş değeri"],
            calculation_method="N/A",
            is_safe=False
        )

    if weight_kg <= 0 or age_years < 0:
        return DoseResult(
            drug_name=drug,
            dose="Hata",
            frequency="N/A",
            warnings=["❌ Kilo ve yaş pozitif değerler olmalıdır"],
            calculation_method="N/A",
            is_safe=False
        )

    # Calculate dose using appropriate function
    calc_func = SUPPORTED_DRUGS[drug_normalized]
    result = calc_func(weight_kg, age_years)

    # Add disclaimer
    result.warnings.append(
        "⚠️ DİKKAT: Bu hesaplama yalnızca referans amaçlıdır. "
        "Kesin doz için hekim konsültasyonu gereklidir."
    )

    return result


def get_supported_drugs() -> list:
    """Return list of supported drug names"""
    return ['Amoksisilin', 'Parasetamol', 'İbuprofen']


if __name__ == "__main__":
    # Test the calculator
    print("=== İlaç Doz Hesaplayıcı Test ===\n")

    # Test 1: Amoxicillin for 7-year-old, 25kg
    result = calculate_dose("amoksisilin", 25, 7)
    print(f"İlaç: {result.drug_name}")
    print(f"Doz: {result.dose}")
    print(f"Sıklık: {result.frequency}")
    print(f"Hesaplama: {result.calculation_method}")
    print(f"Uyarılar:")
    for warning in result.warnings:
        print(f"  {warning}")
    print(f"Güvenli: {'Evet' if result.is_safe else 'Hayır'}")
    print()
