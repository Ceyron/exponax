# Repeated Stepper

Use this to create steppers that perform substepping. To do this instantiate those with a subset of the desired `dt`. For example,

```python
substepped_stepper = exponax.stepper.Burgers(1, 1.0, 64, 0.1/5)
stepper = exponax.stepper.RepeatedStepper(substepped_stepper, 5)
```

This will create a stepper that performs 5 substeps of 0.1/5=0.02 each time it is called.

::: exponax.RepeatedStepper
    options:
        members:
            - __init__
            - __call__