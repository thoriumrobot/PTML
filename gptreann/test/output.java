public class DataUtil {
public static String formatData(@Nullable String data) {
		return "Formatted: " + data.toUpperCase();
	}

	public static void main(String[] args) {
		String result = formatData(null);
	}
}
