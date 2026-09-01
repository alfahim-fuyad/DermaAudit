from django import template


register = template.Library()


@register.filter
def percent(value):
    """Convert stored probability fractions to display percentages."""
    try:
        return round(float(value) * 100, 2)
    except (TypeError, ValueError):
        return 0