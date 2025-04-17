# XIRR Calculator

This Python project provides tools to calculate the Extended Internal Rate of Return (XIRR) for investments with irregular cash flows. XIRR is particularly useful for assessing the annualized return of investments where transactions occur at non-periodic intervals.

## Features

- **Flexible Cash Flow Analysis**: Calculate XIRR for single or multiple investment periods.
- **Multiple Calculation Scripts**:
  - `xirr.py`: Core XIRR calculation logic.
  - `xirr_decade.py`: Analyze XIRR over a decade.
  - `xirr_multi_year.py`: Evaluate XIRR across multiple years.

## Prerequisites

Ensure you have Python installed. Required dependencies are listed in `requirement.txt`.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/kunwarmahen/xirr.git
   cd xirr
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirement.txt
   ```

## Usage

### 1. Calculate XIRR for a Single Investment

Run the core script with your cash flow data:

```bash
python xirr.py
```

Ensure your cash flow data is formatted appropriately within the script or modify it to read from an external source.

### 2. Analyze XIRR Over a Decade

Use the decade analysis script:

```bash
python xirr_decade.py
```

This script is designed to process and calculate XIRR for a ten-year investment period.

### 3. Evaluate XIRR Across Multiple Years

For multi-year analysis:

```bash
python xirr_multi_year.py
```

This allows for assessment of XIRR over various investment durations.

### 4. Sample Outputs

![decade_comparison](https://github.com/user-attachments/assets/c9f9ccde-9b7e-40d3-80db-f6315ab34cd5)

![xirr_comparison_periods](https://github.com/user-attachments/assets/de3b2e6d-06d5-4b21-b7e8-8acc25dd4f37)

![xirr_trends](https://github.com/user-attachments/assets/e5f03c21-4b9e-44e5-a8ea-70e31421b800)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

## License

This project is open-source and available under the [MIT License].

## Acknowledgments

Inspired by the need for accurate financial analysis tools to assess investment performance over irregular intervals.
